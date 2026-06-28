use std::{collections::HashSet, net::SocketAddr, path::PathBuf, sync::Arc};

use axum::{
    extract::{Form, Path, State},
    http::{header, HeaderMap, StatusCode},
    response::{Html, IntoResponse, Redirect, Response},
    routing::{get, post},
    Router,
};
use rand::{distributions::Alphanumeric, Rng};
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::{agent::session, state::BotState};

#[derive(Clone)]
pub struct AdminConfig {
    pub addr: String,
    pub username: String,
    pub password: String,
}

#[derive(Clone)]
struct AdminState {
    bot: BotState,
    cfg: AdminConfig,
    sessions: Arc<Mutex<HashSet<String>>>,
}

#[derive(Debug, Deserialize)]
struct LoginForm {
    username: String,
    password: String,
}

#[derive(Debug, Deserialize)]
struct ActionForm {
    action: String,
}

pub fn spawn(bot: BotState, cfg: AdminConfig) {
    tokio::spawn(async move {
        if let Err(e) = serve(bot, cfg).await {
            tracing::error!("admin web failed: {e:#}");
        }
    });
}

async fn serve(bot: BotState, cfg: AdminConfig) -> anyhow::Result<()> {
    let addr: SocketAddr = cfg.addr.parse()?;
    let state = AdminState {
        bot,
        cfg,
        sessions: Arc::new(Mutex::new(HashSet::new())),
    };
    let app = Router::new()
        .route("/", get(|| async { Redirect::to("/admin/users") }))
        .route("/admin", get(|| async { Redirect::to("/admin/users") }))
        .route("/admin/login", get(login_page).post(login))
        .route("/admin/logout", post(logout))
        .route("/admin/users", get(users_page))
        .route("/admin/users/:chat_id", get(user_page))
        .route("/admin/users/:chat_id/raw", get(user_raw))
        .route("/admin/users/:chat_id/action", post(user_action))
        .with_state(state);
    tracing::info!("Admin web listening on http://{addr}/admin");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn login_page() -> Html<String> {
    Html(layout(
        "Login",
        r#"
        <section class="panel narrow">
          <h1>tg-agent admin</h1>
          <form method="post" action="/admin/login" class="stack">
            <label>Login <input name="username" autocomplete="username"></label>
            <label>Password <input name="password" type="password" autocomplete="current-password"></label>
            <button type="submit">Sign in</button>
          </form>
        </section>
        "#,
    ))
}

async fn login(State(st): State<AdminState>, Form(form): Form<LoginForm>) -> Response {
    if form.username == st.cfg.username && form.password == st.cfg.password {
        let token: String = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(48)
            .map(char::from)
            .collect();
        st.sessions.lock().await.insert(token.clone());
        (
            [
                (
                    header::SET_COOKIE,
                    format!("admin_session={token}; HttpOnly; SameSite=Lax; Path=/admin"),
                ),
                (header::LOCATION, "/admin/users".to_string()),
            ],
            StatusCode::SEE_OTHER,
        )
            .into_response()
    } else {
        Html(layout(
            "Login failed",
            r#"<section class="panel narrow"><h1>Login failed</h1><p>Bad login or password.</p><a href="/admin/login">Try again</a></section>"#,
        ))
        .into_response()
    }
}

async fn logout(State(st): State<AdminState>, headers: HeaderMap) -> Response {
    if let Some(token) = cookie_token(&headers) {
        st.sessions.lock().await.remove(&token);
    }
    (
        [(
            header::SET_COOKIE,
            "admin_session=; Max-Age=0; Path=/admin".to_string(),
        )],
        Redirect::to("/admin/login"),
    )
        .into_response()
}

async fn users_page(State(st): State<AdminState>, headers: HeaderMap) -> Response {
    if let Some(resp) = require_admin(&st, &headers).await {
        return resp;
    }
    let access = st.bot.access_snapshot().await;
    let mut ids: HashSet<i64> = access.authorized_chat_ids.iter().copied().collect();
    if let Some(root) = access.root_chat_id {
        ids.insert(root);
    }
    for s in list_sessions() {
        ids.insert(s.chat_id);
    }
    let mut ids: Vec<i64> = ids.into_iter().collect();
    ids.sort();

    let mut rows = String::new();
    for id in ids {
        let s = session::load(id);
        let role = if access.root_chat_id == Some(id) {
            "root"
        } else if access.authorized_chat_ids.contains(&id) {
            "user"
        } else {
            "locked"
        };
        rows.push_str(&format!(
            "<tr><td><a href=\"/admin/users/{id}\">{id}</a></td><td>{role}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            s.profile.fields.len(),
            s.notes.entries.len(),
            s.memory.facts.len(),
            s.memory.recent.len()
        ));
    }

    Html(layout(
        "Users",
        &format!(
            r#"
            <div class="topbar">
              <h1>Users</h1>
              <form method="post" action="/admin/logout"><button>Logout</button></form>
            </div>
            <section class="panel">
              <table>
                <thead><tr><th>Chat ID</th><th>Role</th><th>Profile</th><th>Notes</th><th>Facts</th><th>Messages</th></tr></thead>
                <tbody>{rows}</tbody>
              </table>
            </section>
            "#
        ),
    ))
    .into_response()
}

async fn user_page(
    State(st): State<AdminState>,
    headers: HeaderMap,
    Path(chat_id): Path<i64>,
) -> Response {
    if let Some(resp) = require_admin(&st, &headers).await {
        return resp;
    }
    let s = session::load(chat_id);
    let access = st.bot.access_snapshot().await;
    let role = if access.root_chat_id == Some(chat_id) {
        "root"
    } else if access.authorized_chat_ids.contains(&chat_id) {
        "user"
    } else {
        "locked"
    };
    let watches = st.bot.list_watches_for_chat(chat_id).await;
    let push_subs: Vec<_> = st
        .bot
        .push_subs
        .lock()
        .await
        .iter()
        .filter(|p| p.chat_id == chat_id)
        .cloned()
        .collect();

    let profile = lines_or_empty(s.profile.render_lines());
    let notes = lines_or_empty(s.notes.render_lines());
    let facts = lines_or_empty(
        s.memory
            .facts
            .iter()
            .map(|f| format!("{} [{}]: {}", f.key, f.layer, f.value))
            .collect(),
    );
    let recent = if s.memory.recent.is_empty() {
        "<p class=\"muted\">empty</p>".to_string()
    } else {
        s.memory
            .recent
            .iter()
            .enumerate()
            .map(|(i, (role, text))| {
                format!(
                    "<article class=\"msg\"><b>{}. {}</b><pre>{}</pre></article>",
                    i + 1,
                    esc(role),
                    esc(text)
                )
            })
            .collect::<Vec<_>>()
            .join("")
    };
    let watch_lines = if watches.is_empty() {
        "empty".to_string()
    } else {
        watches
            .iter()
            .map(|w| {
                format!(
                    "#{} {}/{} every {}m",
                    w.id, w.server, w.tool, w.interval_min
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let push_lines = if push_subs.is_empty() {
        "empty".to_string()
    } else {
        push_subs
            .iter()
            .map(|p| format!("{} period={}", p.server, p.period))
            .collect::<Vec<_>>()
            .join("\n")
    };

    Html(layout(
        &format!("User {chat_id}"),
        &format!(
            r#"
            <div class="topbar">
              <div><a href="/admin/users">← Users</a><h1>User {chat_id}</h1><p class="muted">Role: {role}</p></div>
              <a class="button" href="/admin/users/{chat_id}/raw">Raw JSON</a>
            </div>
            <section class="grid">
              <div class="panel"><h2>Profile</h2><pre>{profile}</pre></div>
              <div class="panel"><h2>Notes</h2><pre>{notes}</pre></div>
              <div class="panel"><h2>Facts</h2><pre>{facts}</pre></div>
              <div class="panel"><h2>Summary</h2><pre>{}</pre></div>
              <div class="panel"><h2>Watches</h2><pre>{}</pre></div>
              <div class="panel"><h2>Push subscriptions</h2><pre>{}</pre></div>
            </section>
            <section class="panel">
              <h2>Recent messages</h2>
              {recent}
            </section>
            <section class="panel danger">
              <h2>Manage</h2>
              <div class="actions">
                {}
              </div>
            </section>
            "#,
            esc(&s.memory.summary),
            esc(&watch_lines),
            esc(&push_lines),
            action_buttons(chat_id, role),
        ),
    ))
    .into_response()
}

async fn user_raw(
    State(st): State<AdminState>,
    headers: HeaderMap,
    Path(chat_id): Path<i64>,
) -> Response {
    if let Some(resp) = require_admin(&st, &headers).await {
        return resp;
    }
    let s = session::load(chat_id);
    let raw = serde_json::to_string_pretty(&s).unwrap_or_else(|_| "{}".into());
    Html(layout(
        &format!("Raw {chat_id}"),
        &format!(
            r#"<p><a href="/admin/users/{chat_id}">← User</a></p><section class="panel"><pre>{}</pre></section>"#,
            esc(&raw)
        ),
    ))
    .into_response()
}

async fn user_action(
    State(st): State<AdminState>,
    headers: HeaderMap,
    Path(chat_id): Path<i64>,
    Form(form): Form<ActionForm>,
) -> Response {
    if let Some(resp) = require_admin(&st, &headers).await {
        return resp;
    }
    let mut s = session::load(chat_id);
    match form.action.as_str() {
        "grant_access" => st.bot.grant_access(chat_id).await,
        "revoke_access" => {
            st.bot.revoke_access(chat_id).await;
        }
        "make_root" => st.bot.set_root(chat_id).await,
        "compact" => {
            s.memory.clear_short_context();
            let _ = session::save(&s);
        }
        "reset_memory" => {
            s.memory.reset_for_new_session();
            s.trip = None;
            let _ = session::save(&s);
        }
        "clear_profile" => {
            s.profile.clear();
            let _ = session::save(&s);
        }
        "clear_notes" => {
            s.notes.clear();
            let _ = session::save(&s);
        }
        "delete_session" => {
            let _ = delete_session_file(chat_id);
        }
        "stop_watches" => {
            let servers: Vec<String> = st
                .bot
                .push_subs
                .lock()
                .await
                .iter()
                .filter(|p| p.chat_id == chat_id)
                .map(|p| p.server.clone())
                .collect();
            for server in &servers {
                let mut args = serde_json::Map::new();
                args.insert("session_id".into(), chat_id.to_string().into());
                let _ = st
                    .bot
                    .call_tool(server, "unsubscribe_summaries", Some(args))
                    .await;
            }
            st.bot.remove_watches_for_chat(chat_id).await;
            st.bot.remove_push_subs(chat_id, None).await;
        }
        _ => {}
    }
    Redirect::to(&format!("/admin/users/{chat_id}")).into_response()
}

async fn require_admin(st: &AdminState, headers: &HeaderMap) -> Option<Response> {
    let Some(token) = cookie_token(headers) else {
        return Some(Redirect::to("/admin/login").into_response());
    };
    if st.sessions.lock().await.contains(&token) {
        None
    } else {
        Some(Redirect::to("/admin/login").into_response())
    }
}

fn cookie_token(headers: &HeaderMap) -> Option<String> {
    let cookie = headers.get(header::COOKIE)?.to_str().ok()?;
    for part in cookie.split(';') {
        let Some((k, v)) = part.trim().split_once('=') else {
            continue;
        };
        if k == "admin_session" && !v.is_empty() {
            return Some(v.to_string());
        }
    }
    None
}

#[derive(Debug)]
struct SessionRow {
    chat_id: i64,
}

fn list_sessions() -> Vec<SessionRow> {
    let dir = session::sessions_dir();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Ok(chat_id) = stem.parse::<i64>() {
                out.push(SessionRow { chat_id });
            }
        }
    }
    out.sort_by_key(|s| s.chat_id);
    out
}

fn delete_session_file(chat_id: i64) -> std::io::Result<()> {
    let path: PathBuf = session::sessions_dir().join(format!("{chat_id}.json"));
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

fn action_buttons(chat_id: i64, role: &str) -> String {
    let actions = [
        ("grant_access", "Grant access"),
        ("revoke_access", "Revoke access"),
        ("make_root", "Make root"),
        ("compact", "Clear context"),
        ("reset_memory", "Reset memory"),
        ("clear_profile", "Clear profile"),
        ("clear_notes", "Clear notes"),
        ("stop_watches", "Stop watches"),
        ("delete_session", "Delete session file"),
    ];
    actions
        .iter()
        .filter(|(action, _)| !(role == "root" && *action == "revoke_access"))
        .map(|(action, label)| {
            format!(
                r#"<form method="post" action="/admin/users/{chat_id}/action"><input type="hidden" name="action" value="{action}"><button>{label}</button></form>"#
            )
        })
        .collect::<Vec<_>>()
        .join("")
}

fn lines_or_empty(lines: Vec<String>) -> String {
    if lines.is_empty() {
        "empty".into()
    } else {
        esc(&lines.join("\n"))
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn layout(title: &str, body: &str) -> String {
    format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{}</title>
  <style>
    :root {{ color-scheme: light; --bg:#f6f7f9; --panel:#fff; --line:#d8dee7; --text:#111827; --muted:#657386; --accent:#0f766e; --danger:#b42318; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; background:var(--bg); color:var(--text); }}
    main {{ max-width:1180px; margin:0 auto; padding:24px; }}
    h1 {{ margin:0 0 8px; font-size:28px; }}
    h2 {{ margin:0 0 12px; font-size:18px; }}
    a {{ color:var(--accent); text-decoration:none; }}
    table {{ width:100%; border-collapse:collapse; }}
    th,td {{ text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
    th {{ color:var(--muted); font-weight:600; }}
    input {{ width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:6px; background:#fff; }}
    button,.button {{ display:inline-block; padding:9px 12px; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--text); cursor:pointer; }}
    button:hover,.button:hover {{ border-color:var(--accent); color:var(--accent); }}
    pre {{ margin:0; white-space:pre-wrap; overflow-wrap:anywhere; font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; margin:14px 0; }}
    .narrow {{ max-width:420px; margin:12vh auto; }}
    .stack {{ display:grid; gap:14px; }}
    .topbar {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; margin-bottom:12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
    .grid .panel {{ margin:0; }}
    .muted {{ color:var(--muted); margin:0; }}
    .msg {{ border-top:1px solid var(--line); padding:12px 0; }}
    .msg:first-child {{ border-top:0; }}
    .actions {{ display:flex; flex-wrap:wrap; gap:10px; }}
    .danger button:hover {{ border-color:var(--danger); color:var(--danger); }}
    @media (max-width:760px) {{ main {{ padding:14px; }} .grid {{ grid-template-columns:1fr; }} .topbar {{ display:block; }} }}
  </style>
</head>
<body><main>{}</main></body>
</html>"#,
        esc(title),
        body
    )
}
