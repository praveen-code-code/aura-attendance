
from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user

from models import db, Admin

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("admin.dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        admin = Admin.query.filter_by(email=email).first()
        if admin and admin.check_password(password):
            login_user(admin)
            flash("Logged in successfully.", "success")
            return redirect(url_for("admin.dashboard"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("auth.login"))


@auth_bp.route("/setup", methods=["GET", "POST"])
def setup():
    """One-time bootstrap route to create the first admin account when the
    admins table is empty. Disables itself once an admin exists."""
    if Admin.query.first() is not None:
        flash("Admin already exists. Please log in.", "info")
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or len(password) < 8:
            flash("Name, email, and an 8+ character password are required.", "error")
            return render_template("setup.html")

        admin = Admin(name=name, email=email)
        admin.set_password(password)
        db.session.add(admin)
        db.session.commit()
        flash("Admin account created. Please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("setup.html")
