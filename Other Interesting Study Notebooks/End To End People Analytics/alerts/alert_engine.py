"""
alert_engine.py
----------------
Motor de alertas para empleados en alto riesgo de fuga.
Soporta notificaciones por email (Gmail SMTP) y Slack (Webhook).
Configurable via variables de entorno.
"""

import os
import smtplib
import json
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from typing import List, Dict, Any

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


# --- Configuración desde variables de entorno ---
ALERT_THRESHOLD = float(os.environ.get("ALERT_THRESHOLD", "0.65"))
ALERT_EMAIL_TO = os.environ.get("ALERT_EMAIL_TO", "")           # Destinatario del email
ALERT_EMAIL_FROM = os.environ.get("ALERT_EMAIL_FROM", "")       # Tu cuenta Gmail
ALERT_EMAIL_PASSWORD = os.environ.get("ALERT_EMAIL_PASSWORD", "")  # Contraseña o app password
ALERT_SLACK_WEBHOOK = os.environ.get("ALERT_SLACK_WEBHOOK", "")  # URL del Webhook de Slack


def _build_email_html(high_risk_employees: List[Dict[str, Any]], threshold: float) -> str:
    """Construye el cuerpo HTML del email de alerta."""
    rows = ""
    for emp in high_risk_employees:
        prob = emp.get("flight_risk_prob", 0)
        color = "#FF4B4B" if prob >= 0.75 else "#FF8C00"
        rows += f"""
        <tr>
            <td style="padding:8px;border:1px solid #ddd;">{emp.get('employee_id','')}</td>
            <td style="padding:8px;border:1px solid #ddd;">{emp.get('name','')}</td>
            <td style="padding:8px;border:1px solid #ddd;">{emp.get('role','')}</td>
            <td style="padding:8px;border:1px solid #ddd;color:{color};font-weight:bold;">
                {prob*100:.1f}%
            </td>
            <td style="padding:8px;border:1px solid #ddd;">{emp.get('workload_index', 'N/A')}</td>
            <td style="padding:8px;border:1px solid #ddd;">{emp.get('overtime_hours_last_month', 'N/A')}</td>
        </tr>
        """

    return f"""
    <html><body style="font-family:Arial,sans-serif;color:#333;">
    <div style="max-width:700px;margin:auto;padding:20px;">
        <h2 style="color:#E15759;">⚠️ Alerta de People Analytics — Alto Riesgo de Fuga</h2>
        <p>
            Generado: <b>{datetime.now().strftime('%Y-%m-%d %H:%M')}</b><br>
            Umbral de alerta: <b>{threshold*100:.0f}%</b><br>
            Empleados en alerta: <b>{len(high_risk_employees)}</b>
        </p>
        <table style="border-collapse:collapse;width:100%;">
            <thead>
                <tr style="background:#2D3047;color:white;">
                    <th style="padding:8px;">ID</th>
                    <th style="padding:8px;">Nombre</th>
                    <th style="padding:8px;">Rol</th>
                    <th style="padding:8px;">Riesgo de Fuga</th>
                    <th style="padding:8px;">Carga Trabajo</th>
                    <th style="padding:8px;">Horas Extra</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        <br>
        <p style="font-size:12px;color:#777;">
            Este mensaje fue generado automáticamente por la Suite de People Analytics.
            Accede al dashboard en <a href="http://localhost:8501">http://localhost:8501</a>
            para ver el análisis completo de cada empleado.
        </p>
    </div>
    </body></html>
    """


def _build_slack_payload(high_risk_employees: List[Dict[str, Any]], threshold: float) -> Dict:
    """Construye el payload JSON para la notificación de Slack."""
    employee_lines = []
    for emp in high_risk_employees[:10]:  # Limitamos a 10 en Slack para no saturar
        prob = emp.get("flight_risk_prob", 0) * 100
        icon = "🔴" if prob >= 75 else "🟡"
        employee_lines.append(
            f"{icon} *{emp.get('name', '')}* ({emp.get('role', '')}) — Riesgo: *{prob:.1f}%*"
        )

    if len(high_risk_employees) > 10:
        employee_lines.append(f"_...y {len(high_risk_employees) - 10} colaboradores más_")

    employees_text = "\n".join(employee_lines)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    return {
        "text": f"⚠️ *Alerta People Analytics — {len(high_risk_employees)} empleados en alto riesgo*",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"⚠️ Alerta de Fuga de Talento — {timestamp}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Umbral activo:* {threshold*100:.0f}%  |  "
                        f"*Empleados en riesgo:* {len(high_risk_employees)}\n\n"
                        f"{employees_text}"
                    )
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Ver Dashboard", "emoji": True},
                        "url": "http://localhost:8501",
                        "style": "danger"
                    }
                ]
            }
        ]
    }


def send_email_alert(high_risk_employees: List[Dict[str, Any]], threshold: float = ALERT_THRESHOLD) -> bool:
    """
    Envía un email de alerta via Gmail SMTP.

    Requiere variables de entorno:
        ALERT_EMAIL_FROM, ALERT_EMAIL_PASSWORD, ALERT_EMAIL_TO

    Retorna True si se envió correctamente, False en caso contrario.
    """
    if not all([ALERT_EMAIL_FROM, ALERT_EMAIL_PASSWORD, ALERT_EMAIL_TO]):
        logger.warning(
            "Email alert skipped: ALERT_EMAIL_FROM, ALERT_EMAIL_PASSWORD, ALERT_EMAIL_TO "
            "no están configurados como variables de entorno."
        )
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[People Analytics] ⚠️ {len(high_risk_employees)} empleados en alto riesgo de fuga"
        msg["From"] = ALERT_EMAIL_FROM
        msg["To"] = ALERT_EMAIL_TO

        html_body = _build_email_html(high_risk_employees, threshold)
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(ALERT_EMAIL_FROM, ALERT_EMAIL_PASSWORD)
            server.sendmail(ALERT_EMAIL_FROM, ALERT_EMAIL_TO, msg.as_string())

        logger.info(f"Email de alerta enviado a {ALERT_EMAIL_TO} ({len(high_risk_employees)} empleados)")
        return True

    except Exception as e:
        logger.error(f"Error al enviar email de alerta: {e}")
        return False


def send_slack_alert(high_risk_employees: List[Dict[str, Any]], threshold: float = ALERT_THRESHOLD) -> bool:
    """
    Envía una notificación a un canal de Slack via Incoming Webhook.

    Requiere variable de entorno:
        ALERT_SLACK_WEBHOOK — URL del webhook de Slack

    Retorna True si se envió correctamente, False en caso contrario.
    """
    if not ALERT_SLACK_WEBHOOK:
        logger.warning(
            "Slack alert skipped: ALERT_SLACK_WEBHOOK no está configurado como variable de entorno."
        )
        return False

    try:
        payload = _build_slack_payload(high_risk_employees, threshold)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            ALERT_SLACK_WEBHOOK,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                logger.info(f"Slack alert enviado ({len(high_risk_employees)} empleados)")
                return True
            else:
                logger.error(f"Slack respondió con status {resp.status}")
                return False

    except Exception as e:
        logger.error(f"Error al enviar Slack alert: {e}")
        return False


def fire_alerts(employees_risk: List[Dict[str, Any]], threshold: float = None) -> Dict[str, Any]:
    """
    Función principal del motor de alertas.
    Filtra empleados sobre el umbral y dispara todos los canales configurados.

    Parámetros
    ----------
    employees_risk : List[Dict]
        Lista de empleados con sus datos de riesgo (flight_risk_prob, name, role, etc.)
    threshold : float, opcional
        Umbral de riesgo. Si no se pasa, usa ALERT_THRESHOLD del entorno.

    Retorna
    -------
    Dict con:
        - high_risk_count: número de empleados alertados
        - email_sent: bool
        - slack_sent: bool
        - employees_alerted: lista de IDs de empleados alertados
    """
    if threshold is None:
        threshold = ALERT_THRESHOLD

    # Filtrar empleados sobre el umbral
    high_risk = [
        emp for emp in employees_risk
        if emp.get("flight_risk_prob", 0) >= threshold
    ]

    result = {
        "high_risk_count": len(high_risk),
        "threshold_used": threshold,
        "email_sent": False,
        "slack_sent": False,
        "employees_alerted": [emp.get("employee_id") for emp in high_risk],
    }

    if len(high_risk) == 0:
        logger.info("Motor de alertas: ningún empleado supera el umbral configurado.")
        return result

    logger.info(f"Motor de alertas: {len(high_risk)} empleados sobre umbral {threshold:.0%}")

    # Disparar canales
    result["email_sent"] = send_email_alert(high_risk, threshold)
    result["slack_sent"] = send_slack_alert(high_risk, threshold)

    return result
