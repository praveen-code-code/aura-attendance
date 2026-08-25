
from datetime import date, timedelta
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required
from sqlalchemy import func

from models import db, Student, Attendance
from ml.predictor import predictor

admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/dashboard")
@login_required
def dashboard():
    total_students = Student.query.count()
    today_present = Attendance.query.filter_by(date=date.today(), status="PRESENT").count()

    week_ago = date.today() - timedelta(days=7)
    recent = (
        Attendance.query.filter(Attendance.date >= week_ago)
        .order_by(Attendance.date.desc(), Attendance.time.desc())
        .limit(10)
        .all()
    )

    return render_template(
        "dashboard.html",
        total_students=total_students,
        today_present=today_present,
        models_ready=predictor.is_ready(),
        recent=recent,
    )


@admin_bp.route("/reports")
@login_required
def reports():
    start = request.args.get("start")
    end = request.args.get("end")

    query = Attendance.query.filter_by(status="PRESENT")
    if start:
        query = query.filter(Attendance.date >= start)
    if end:
        query = query.filter(Attendance.date <= end)

    records = query.order_by(Attendance.date.desc()).all()
    return render_template("reports.html", records=records, start=start, end=end)


@admin_bp.route("/api/analytics/daily-counts")
@login_required
def daily_counts():
    """Attendance count per day for the last 14 days (feeds a chart)."""
    since = date.today() - timedelta(days=14)
    rows = (
        db.session.query(Attendance.date, func.count(Attendance.id))
        .filter(Attendance.date >= since, Attendance.status == "PRESENT")
        .group_by(Attendance.date)
        .order_by(Attendance.date)
        .all()
    )
    return jsonify([{"date": d.isoformat(), "count": c} for d, c in rows])
