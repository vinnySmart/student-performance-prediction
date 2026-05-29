"""
seed_users.py — Create default admin and faculty accounts.
Run ONCE after creating the database:
    python seed_users.py
"""
from app import create_app, db, bcrypt
from app.models import User

app = create_app()

with app.app_context():
    db.create_all()

    users = [
        {'username': 'admin',    'password': 'Admin@123', 'role': 'admin'},
        {'username': 'faculty1', 'password': 'Faculty@123', 'role': 'faculty'},
    ]

    for u in users:
        if not User.query.filter_by(username=u['username']).first():
            hashed = bcrypt.generate_password_hash(u['password']).decode('utf-8')
            user = User(username=u['username'],
                        password_hash=hashed, role=u['role'])
            db.session.add(user)
            print(f"Created user: {u['username']} ({u['role']})")
        else:
            print(f"User '{u['username']}' already exists, skipping.")

    db.session.commit()
    print("\nDone. Change these passwords before any deployment!")
