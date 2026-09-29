from unittest.mock import patch

import auth_rate_limit


def test_login_limiter_blocks_after_configured_failures_and_clears_on_success():
    with patch.object(auth_rate_limit, "API_AUTH_MAX_ATTEMPTS", 2), \
         patch.object(auth_rate_limit, "API_AUTH_WINDOW_SECONDS", 60):
        auth_rate_limit.clear_login_failures("127.0.0.1", "alice")
        assert auth_rate_limit.login_allowed("127.0.0.1", "alice")[0]
        auth_rate_limit.record_login_failure("127.0.0.1", "alice")
        auth_rate_limit.record_login_failure("127.0.0.1", "alice")
        allowed, retry_after = auth_rate_limit.login_allowed("127.0.0.1", "alice")
        assert not allowed
        assert retry_after > 0
        auth_rate_limit.clear_login_failures("127.0.0.1", "alice")
        assert auth_rate_limit.login_allowed("127.0.0.1", "alice")[0]


def test_login_checks_do_not_accumulate_unbounded_keys():
    auth_rate_limit._FAILURES.clear()
    # 只是查询（未失败）的探测不应在内存里留下任何条目。
    for index in range(1000):
        auth_rate_limit.login_allowed(f"10.0.0.{index}", f"user-{index}")
    assert len(auth_rate_limit._FAILURES) == 0


def test_expired_failures_are_pruned_from_memory():
    clock = [1000.0]
    with patch.object(auth_rate_limit, "API_AUTH_WINDOW_SECONDS", 60), \
         patch.object(auth_rate_limit.time, "monotonic", lambda: clock[0]):
        auth_rate_limit._FAILURES.clear()
        auth_rate_limit.record_login_failure("127.0.0.1", "bob")
        assert len(auth_rate_limit._FAILURES) == 1
        # 时间推进超过窗口后，过期记录应在下次检查时被清除。
        clock[0] += 61
        auth_rate_limit.login_allowed("127.0.0.1", "bob")
        assert len(auth_rate_limit._FAILURES) == 0


def test_limiter_scopes_do_not_share_a_budget():
    with patch.object(auth_rate_limit, "_FAILURES", {}):
        for _ in range(3):
            auth_rate_limit.record_attempt_failure(auth_rate_limit.EMAIL_VERIFY_SCOPE, "127.0.0.1", "alice")

        assert not auth_rate_limit.attempt_allowed(auth_rate_limit.EMAIL_VERIFY_SCOPE, "127.0.0.1", "alice", 3)[0]
        # The same address and identifier are still free to sign in.
        assert auth_rate_limit.attempt_allowed(auth_rate_limit.LOGIN_SCOPE, "127.0.0.1", "alice", 3)[0]
