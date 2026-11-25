# viba_acp/apis.viba

## `new_session`

```viba
new_session =
     Session
  <- $env Env
```

### Session

```viba
Session =
    Object
  * $id (
        SessionId
      * { the session's id }
    )
  * $title (
        str
      * { what it shows }
    )
```
