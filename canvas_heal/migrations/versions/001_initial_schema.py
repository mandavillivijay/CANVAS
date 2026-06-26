"""Initial canvas-heal schema for PostgreSQL.

Revision ID: 001
Revises:
Create Date: 2026-06-26
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "intents",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("team_id", sa.Text, nullable=False, server_default=""),
        sa.Column("project_id", sa.Text, nullable=False, server_default=""),
        sa.Column("selector", sa.Text, nullable=False),
        sa.Column("descriptor", sa.Text, nullable=False),
        sa.Column("embedding", sa.LargeBinary, nullable=False),
        sa.Column("model_name", sa.Text, nullable=False, server_default=""),
        sa.Column("page_url", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("name", "team_id", "project_id", name="uq_intents_name_ns"),
    )
    op.create_table(
        "intent_versions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("intent_name", sa.Text, nullable=False),
        sa.Column("team_id", sa.Text, nullable=False, server_default=""),
        sa.Column("project_id", sa.Text, nullable=False, server_default=""),
        sa.Column("selector", sa.Text, nullable=False),
        sa.Column("descriptor", sa.Text, nullable=False),
        sa.Column("embedding", sa.LargeBinary, nullable=False),
        sa.Column("model_name", sa.Text, nullable=False, server_default=""),
        sa.Column("page_url", sa.Text, nullable=False, server_default=""),
        sa.Column("recorded_by", sa.Text, nullable=False, server_default=""),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "idx_versions_ns",
        "intent_versions",
        ["intent_name", "team_id", "project_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_versions_ns", table_name="intent_versions")
    op.drop_table("intent_versions")
    op.drop_table("intents")
