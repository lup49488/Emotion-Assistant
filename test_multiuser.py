"""
test_multiuser.py
验证多用户改造的核心承诺：
  1. 不同用户的数据完全隔离（文件、内存状态都不串）
  2. 同一用户的并发请求被正确串行化，不丢数据、不损坏文件
  3. user_id 校验能挡住路径穿越类输入
  4. SessionStore 的 LRU 淘汰确实生效
"""

import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import conftest  # noqa: F401  确保 stub 在导入 chatbot_multiuser 前注册


class MultiUserTestBase(unittest.TestCase):
    """每个测试用临时目录跑，互不污染，结束后自动清理。"""

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp(prefix="chatbot_test_")
        # 动态修改 BASE_DIR/USERS_DIR 指向临时目录，再导入模块
        import chatbot_multiuser as cb
        cb.BASE_DIR = Path(self._tmp_dir)
        cb.USERS_DIR = cb.BASE_DIR / "users"
        cb.USERS_DIR.mkdir(parents=True, exist_ok=True)
        self.cb = cb
        # 每个测试都用全新的 SessionStore，避免跨测试缓存串台
        self.cb.session_store = cb.SessionStore()

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)



# 1. user_id 校验

class TestUserIdValidation(MultiUserTestBase):

    def test_valid_user_id_passes(self):
        self.assertEqual(self.cb.validate_user_id("alice_123"), "alice_123")

    def test_path_traversal_attempt_rejected(self):
        with self.assertRaises(ValueError):
            self.cb.validate_user_id("../../etc/passwd")

    def test_user_id_with_slash_rejected(self):
        with self.assertRaises(ValueError):
            self.cb.validate_user_id("alice/bob")

    def test_empty_user_id_rejected(self):
        with self.assertRaises(ValueError):
            self.cb.validate_user_id("")

    def test_overlong_user_id_rejected(self):
        with self.assertRaises(ValueError):
            self.cb.validate_user_id("a" * 200)

    def test_non_string_user_id_rejected(self):
        with self.assertRaises(ValueError):
            self.cb.validate_user_id(12345)  # type: ignore[arg-type]

# 2. 用户数据隔离

class TestUserIsolation(MultiUserTestBase):

    def test_each_user_gets_separate_directory(self):
        alice_dir = self.cb.user_dir("alice")
        bob_dir = self.cb.user_dir("bob")
        self.assertNotEqual(alice_dir, bob_dir)
        self.assertTrue(alice_dir.exists())
        self.assertTrue(bob_dir.exists())

    def test_user_paths_do_not_collide(self):
        alice_paths = self.cb.user_paths("alice")
        bob_paths = self.cb.user_paths("bob")
        for key in alice_paths:
            self.assertNotEqual(alice_paths[key], bob_paths[key])

    def test_interest_saved_by_one_user_not_visible_to_another(self):
        with self.cb.session_store.session("alice") as state:
            self.cb.save_interest({"text": "我喜欢爬山"}, state)

        with self.cb.session_store.session("bob") as state:
            self.assertFalse(state.interest_store.exact_exists("我喜欢爬山"))

    def test_history_is_independent_per_user(self):
        with self.cb.session_store.session("alice") as state:
            self.cb.update_short_term(state, "你好", "你好，alice")

        with self.cb.session_store.session("bob") as state:
            self.assertEqual(state.history, [])

    def test_persisted_files_are_separate_on_disk(self):
        with self.cb.session_store.session("alice") as state:
            self.cb.update_short_term(state, "嗨", "嗨，alice")

        alice_history_path = self.cb.user_paths("alice")["history"]
        bob_history_path = self.cb.user_paths("bob")["history"]
        self.assertTrue(alice_history_path.exists())
        self.assertFalse(bob_history_path.exists())

# 3. 会话持久化与重新加载

class TestPersistenceRoundTrip(MultiUserTestBase):

    def test_state_survives_cache_eviction(self):
        with self.cb.session_store.session("alice") as state:
            self.cb.update_short_term(state, "记住这句话", "好的，我记住了")

        # 模拟缓存被清空（比如服务重启），重新从磁盘加载
        self.cb.session_store = self.cb.SessionStore()
        with self.cb.session_store.session("alice") as state:
            self.assertEqual(len(state.history), 2)
            self.assertEqual(state.history[0]["content"], "记住这句话")

    def test_interest_memory_persists_across_sessions(self):
        with self.cb.session_store.session("alice") as state:
            self.cb.save_interest({"text": "我喜欢弹吉他"}, state)

        self.cb.session_store = self.cb.SessionStore()
        with self.cb.session_store.session("alice") as state:
            self.assertTrue(state.interest_store.exact_exists("我喜欢弹吉他"))

    def test_stable_profile_persists_across_sessions(self):
        with self.cb.session_store.session("alice") as state:
            self.cb.update_stable_profile(state, {"text": "我是学生", "key": "identity"})

        self.cb.session_store = self.cb.SessionStore()
        with self.cb.session_store.session("alice") as state:
            self.assertEqual(state.stable_profile[0]["text"], "我是学生")

    def test_legacy_profile_is_migrated_out_of_long_memory(self):
        paths = self.cb.user_paths("alice")
        self.cb.save_json(paths["long_memory"], [
            {"text": "我是学生", "kind": "profile", "time": "2026-01-01T00:00:00"}
        ])

        with self.cb.session_store.session("alice") as state:
            self.assertEqual(state.long_memory, [])
            self.assertEqual(state.stable_profile[0]["text"], "我是学生")

# 4. 并发安全：同一用户的并发请求

class TestConcurrency(MultiUserTestBase):

    def test_concurrent_writes_same_user_do_not_lose_data(self):
        """
        20 个线程同时对同一用户追加一条短期记忆。
        若锁失效，常见故障模式是部分写入丢失或文件损坏（JSONDecodeError）。
        """
        N = 20
        errors: list[Exception] = []

        def worker(i: int):
            try:
                with self.cb.session_store.session("alice") as state:
                    self.cb.update_short_term(state, f"消息{i}", f"回复{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"并发写入出现异常: {errors}")

        # 文件应该是合法 JSON，且不应该因为竞态而损坏
        history_path = self.cb.user_paths("alice")["history"]
        data = self.cb.load_json(history_path)
        self.assertIsInstance(data, list)
        # SHORT_TERM_LIMIT=10，每次 update 加 2 条，最终应保留最近 10 条
        self.assertLessEqual(len(data), self.cb.SHORT_TERM_LIMIT)

    def test_concurrent_different_users_do_not_block_each_other_results(self):
        """
        不同用户并发写入，最终各自的数据应该完整且互不混淆。
        """
        results: dict[str, list] = {}
        lock = threading.Lock()

        def worker(user_id: str):
            with self.cb.session_store.session(user_id) as state:
                self.cb.update_short_term(state, f"来自{user_id}", f"回复{user_id}")
            with lock:
                results[user_id] = self.cb.load_json(self.cb.user_paths(user_id)["history"])

        users = [f"user{i}" for i in range(10)]
        threads = [threading.Thread(target=worker, args=(u,)) for u in users]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for u in users:
            self.assertEqual(len(results[u]), 2)
            self.assertEqual(results[u][0]["content"], f"来自{u}")

    def test_concurrent_interest_saves_no_duplicate_loss(self):
        """
        同一用户并发保存不同的兴趣记忆，最终应该全部保留（不因竞态丢失）。
        """
        N = 15

        def worker(i: int):
            with self.cb.session_store.session("alice") as state:
                self.cb.save_interest({"text": f"兴趣条目{i}"}, state)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with self.cb.session_store.session("alice") as state:
            self.assertEqual(len(state.interest_store), N)
            for i in range(N):
                self.assertTrue(state.interest_store.exact_exists(f"兴趣条目{i}"))


# ═══════════════════════════════════════════════════════════
# 5. SessionStore 缓存淘汰
# ═══════════════════════════════════════════════════════════

class TestSessionStoreEviction(MultiUserTestBase):

    def test_eviction_keeps_cache_size_bounded(self):
        store = self.cb.SessionStore(max_sessions=3)
        for i in range(5):
            with store.session(f"user{i}"):
                pass
        self.assertLessEqual(len(store._sessions), 3)

    def test_evicted_session_reloads_correctly_from_disk(self):
        store = self.cb.SessionStore(max_sessions=2)

        with store.session("alice") as state:
            self.cb.update_short_term(state, "第一条", "回复")

        # 加载 bob、carol，容量为 2，alice 应被淘汰出缓存
        with store.session("bob"):
            pass
        with store.session("carol"):
            pass
        self.assertNotIn("alice", store._sessions)

        # 但磁盘数据应该还在，重新访问能拿回来
        with store.session("alice") as state:
            self.assertEqual(len(state.history), 2)

    def test_active_session_is_not_evicted(self):
        store = self.cb.SessionStore(max_sessions=1)

        with store.session("alice") as alice_state:
            with store.session("bob"):
                pass
            self.assertIn("alice", store._sessions)
            self.assertIs(store._sessions["alice"], alice_state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
