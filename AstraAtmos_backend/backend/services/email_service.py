"""
邮件发送服务 — 使用 QQ 邮箱 SMTP 发送验证码
"""

import logging
import random
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import (
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM_NAME,
    VERIFICATION_CODE_EXPIRE_MINUTES, VERIFICATION_CODE_COOLDOWN_SECONDS,
)

logger = logging.getLogger("astraatmos.email")

# 内存缓存：{ email: { code, expires_at, sent_at, purpose } }
_code_store: dict[str, dict] = {}
_SMTP_USER_PLACEHOLDERS = {
    "your-email@example.com",
    "example@example.com",
}
_SMTP_PASS_PLACEHOLDERS = {
    "your-email-app-password",
    "your-app-password",
}


def _generate_code() -> str:
    return f"{random.randint(100000, 999999)}"


def _clean_expired():
    now = time.time()
    expired = [k for k, v in _code_store.items() if v["expires_at"] < now]
    for k in expired:
        del _code_store[k]


def _is_smtp_configured() -> bool:
    user = (SMTP_USER or "").strip().strip('"').strip("'").lower()
    pwd = (SMTP_PASSWORD or "").strip().strip('"').strip("'")
    if not user or not pwd:
        return False
    if user in _SMTP_USER_PLACEHOLDERS or "example.com" in user:
        return False
    if pwd.lower() in _SMTP_PASS_PLACEHOLDERS or "your-email" in pwd.lower():
        return False
    return True


def send_verification_code(email: str, purpose: str = "register") -> dict:
    """
    发送验证码到指定邮箱。

    Args:
        email:   目标邮箱
        purpose: "register" | "reset_password"

    Returns:
        {"success": bool, "message": str}
    """
    _clean_expired()
    email = email.lower().strip()
    now = time.time()

    # 冷却检查
    if email in _code_store:
        elapsed = now - _code_store[email]["sent_at"]
        if elapsed < VERIFICATION_CODE_COOLDOWN_SECONDS:
            remaining = int(VERIFICATION_CODE_COOLDOWN_SECONDS - elapsed)
            return {"success": False, "reason": "cooldown", "message": f"请 {remaining} 秒后再试"}

    code = _generate_code()

    # 开发模式：SMTP 未配置时打印到日志
    if not _is_smtp_configured():
        _code_store[email] = {
            "code": code,
            "expires_at": now + VERIFICATION_CODE_EXPIRE_MINUTES * 60,
            "sent_at": now,
            "purpose": purpose,
        }
        logger.warning("[DEV] 验证码未通过邮件发送（SMTP 未配置）: %s -> %s", email, code)
        return {"success": True, "reason": "dev_log", "message": "验证码已发送（本地模式：请查看后端日志）"}

    purpose_text = "注册账号" if purpose == "register" else "重置密码"
    subject = f"【AstraAtmos】{purpose_text}验证码"
    html_body = f"""
    <div style="max-width:480px;margin:0 auto;font-family:'Segoe UI',sans-serif;color:#1a1a2e;">
      <div style="background:linear-gradient(135deg,#c75b39,#e8845a);padding:24px 32px;border-radius:12px 12px 0 0;">
        <h2 style="margin:0;color:#fff;font-size:18px;letter-spacing:2px;">ASTRAATMOS 行星大气实验室</h2>
        <p style="margin:4px 0 0;color:rgba(255,255,255,0.8);font-size:12px;">Planetary Atmosphere Experiment Platform</p>
      </div>
      <div style="background:#ffffff;padding:32px;border:1px solid #eee;border-top:none;border-radius:0 0 12px 12px;">
        <p style="font-size:14px;color:#333;">您正在进行<strong>{purpose_text}</strong>操作，验证码为：</p>
        <div style="text-align:center;margin:24px 0;">
          <span style="display:inline-block;font-size:32px;font-weight:700;letter-spacing:8px;color:#c75b39;
                       background:#fef5f2;padding:12px 28px;border-radius:8px;border:1px dashed #c75b39;">
            {code}
          </span>
        </div>
        <p style="font-size:13px;color:#666;">验证码 {VERIFICATION_CODE_EXPIRE_MINUTES} 分钟内有效，请勿泄露给他人。</p>
        <p style="font-size:12px;color:#999;margin-top:24px;border-top:1px solid #eee;padding-top:16px;">
          如非本人操作，请忽略此邮件。
        </p>
      </div>
    </div>
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = email
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        _code_store[email] = {
            "code": code,
            "expires_at": now + VERIFICATION_CODE_EXPIRE_MINUTES * 60,
            "sent_at": now,
            "purpose": purpose,
        }
        logger.info("验证码已发送: %s (purpose=%s)", email, purpose)
        return {"success": True, "reason": "ok", "message": "验证码已发送，请查看邮箱"}

    except Exception as exc:
        logger.error("邮件发送失败: %s -> %s", email, exc)
        return {"success": False, "reason": "delivery_failed", "message": "邮件发送失败，请检查 SMTP 配置或网络连接"}


def verify_code(email: str, code: str, purpose: str = "register") -> bool:
    """
    校验验证码是否正确且未过期。验证成功后自动删除（一次性）。
    """
    _clean_expired()
    email = email.lower().strip()

    record = _code_store.get(email)
    if not record:
        return False
    if record["purpose"] != purpose:
        return False
    if time.time() > record["expires_at"]:
        del _code_store[email]
        return False
    if record["code"] != code.strip():
        return False

    del _code_store[email]
    return True
