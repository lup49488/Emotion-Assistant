# Frontend Styles

`App.css` is the only stylesheet imported by the application. It preserves a
deliberate cascade order while keeping ownership close to the affected UI:

- `foundation.css`: tokens, controls, forms, and baseline layout rules.
- `workspace.css`: desktop workspace shell, navigation rail, topbar, and themes.
- `feature-pages.css`: personal data, mood, knowledge, privacy, and operations.
- `chat.css`: messages, composer, chat header, settings, and reply context.
- `responsive-navigation.css`: conversation sidebar and responsive/mobile rules.

Add a new rule to the most specific module. Keep its desktop rule with that
component, and place its breakpoint override in the same module when possible.
Shared tokens belong in `foundation.css`; do not add end-of-file overrides to
`App.css`.
