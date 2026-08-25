
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required

from models import db, Student

students_bp = Blueprint("students", __name__)


@students_bp.route("/students")
@login_required
def list_students():
    students = Student.query.order_by(Student.created_at.desc()).all()
    return render_template("students.html", students=students)


@students_bp.route("/students/register", methods=["POST"])
@login_required
def register():
    student_id = request.form.get("student_id", "").strip()
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip() or None
    department = request.form.get("department", "").strip() or None

    if not student_id or not name:
        flash("Student ID and name are required.", "error")
        return redirect(url_for("students.list_students"))

    if Student.query.filter_by(student_id=student_id).first():
        flash(f"Student ID {student_id} is already registered.", "error")
        return redirect(url_for("students.list_students"))

    student = Student(student_id=student_id, name=name, email=email, department=department)
    db.session.add(student)
    db.session.commit()
    flash(f"Registered {name} ({student_id}). Train/retrain models once biometric samples are added.", "success")
    return redirect(url_for("students.list_students"))


@students_bp.route("/students/<int:student_pk>/delete", methods=["POST"])
@login_required
def delete(student_pk):
    student = Student.query.get_or_404(student_pk)
    db.session.delete(student)
    db.session.commit()
    flash(f"Removed {student.name}.", "info")
    return redirect(url_for("students.list_students"))


@students_bp.route("/api/students")
@login_required
def api_list_students():
    students = Student.query.order_by(Student.name).all()
    return jsonify([
        {"student_id": s.student_id, "name": s.name, "department": s.department}
        for s in students
    ])
