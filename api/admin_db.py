"""
后台数据库管理模块
- 管理员账号表
- 全局 LLM 配置表
"""
import sqlite3
import os
import hashlib
import secrets
import time

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'admin.db')

# 初始管理员凭证（可后台修改）
DEFAULT_ADMIN = 'admin'
DEFAULT_PASSWORD = 'xbelievers2026'


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库和默认管理员"""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS llm_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            provider TEXT NOT NULL DEFAULT '',
            api_key TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            extra_requirement TEXT NOT NULL DEFAULT '',
            updated_at INTEGER DEFAULT (strftime('%s', 'now'))
        )
    ''')
    conn.commit()

    # 确保默认管理员存在
    pw_hash = hash_password(DEFAULT_PASSWORD)
    cur.execute('SELECT id FROM admin WHERE username = ?', (DEFAULT_ADMIN,))
    if not cur.fetchone():
        cur.execute('INSERT INTO admin (username, password_hash) VALUES (?, ?)',
                    (DEFAULT_ADMIN, pw_hash))
        conn.commit()

    # 确保 LLM 配置行存在
    cur.execute('SELECT id FROM llm_config WHERE id = 1')
    if not cur.fetchone():
        cur.execute('INSERT INTO llm_config (id) VALUES (1)')
        conn.commit()

    conn.close()


# ─── 密码哈希 ───────────────────────────────────────────

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}${h.hex()}"

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, _ = stored_hash.split('$')
        h = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return h.hex() == stored_hash.split('$')[1]
    except Exception:
        return False

# ─── 管理员操作 ──────────────────────────────────────────

def check_admin(username: str, password: str) -> bool:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('SELECT password_hash FROM admin WHERE username = ?', (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    return verify_password(password, row['password_hash'])

def change_password(username: str, new_password: str) -> bool:
    pw_hash = hash_password(new_password)
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('UPDATE admin SET password_hash = ? WHERE username = ?',
                (pw_hash, username))
    conn.commit()
    affected = cur.rowcount
    conn.close()
    return affected > 0

def get_all_admins():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('SELECT id, username, created_at FROM admin ORDER BY id')
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ─── LLM 配置操作 ─────────────────────────────────────────

def get_llm_config() -> dict:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('SELECT provider, api_key, model, extra_requirement FROM llm_config WHERE id = 1')
    row = cur.fetchone()
    conn.close()
    if row:
        return dict(row)
    return {'provider': '', 'api_key': '', 'model': '', 'extra_requirement': ''}

def save_llm_config(provider: str, api_key: str, model: str, extra_requirement: str = '') -> bool:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute('''
        UPDATE llm_config
        SET provider = ?, api_key = ?, model = ?,
            extra_requirement = ?, updated_at = strftime('%s', 'now')
        WHERE id = 1
    ''', (provider, api_key, model, extra_requirement))
    conn.commit()
    affected = cur.rowcount
    conn.close()
    return affected > 0


# ─── 初始化 ───────────────────────────────────────────────
init_db()
