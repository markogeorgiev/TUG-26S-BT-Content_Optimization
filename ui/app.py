from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

from retriever import run_hybrid_rankings as ranking_backend
from ui.services.ranking_ui_service import (
    DEFAULT_DISPLAY_LIMIT,
    available_display_limits,
    autocomplete_queries,
    execute_or_load_query,
    get_query_record,
    load_query_results,
    recent_queries,
)


def _resolve_display_limit_from_request(source: str = "display_limit") -> str:
    selected_value = request.values.get(source, str(DEFAULT_DISPLAY_LIMIT)).strip()
    if selected_value == "custom":
        custom_value = request.values.get("display_limit_custom", "").strip()
        return custom_value or str(DEFAULT_DISPLAY_LIMIT)
    return selected_value or str(DEFAULT_DISPLAY_LIMIT)


def _display_selection_state(display_limit: int, display_options: list[int]) -> tuple[str, str]:
    if display_limit in display_options:
        return str(display_limit), ""
    return "custom", str(display_limit)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    @app.get("/")
    def index():
        recent = recent_queries()
        display_options = available_display_limits(0)
        selected_display_value, custom_display_limit = _display_selection_state(
            DEFAULT_DISPLAY_LIMIT,
            display_options,
        )
        return render_template(
            "search.html",
            current_query="",
            query_record=None,
            results=[],
            total_results=0,
            display_limit=DEFAULT_DISPLAY_LIMIT,
            display_options=display_options,
            selected_display_value=selected_display_value,
            custom_display_limit=custom_display_limit,
            recent_queries=recent,
            created_state=None,
            error_message=request.args.get("error", ""),
        )

    @app.post("/search")
    def search():
        query = request.form.get("query", "").strip()
        if not query:
            return redirect(url_for("index", error="Please enter a query."))

        display_limit_raw = _resolve_display_limit_from_request("display_limit")
        record, _, created = execute_or_load_query(query)
        return redirect(
            url_for(
                "query_results",
                query_id=record["query_id"],
                display=display_limit_raw,
                state="created" if created else "reused",
            )
        )

    @app.get("/queries/<query_id>")
    def query_results(query_id: str):
        try:
            display_requested = request.args.get("display", str(DEFAULT_DISPLAY_LIMIT)).strip()
            record, visible_results, total_results = load_query_results(
                query_id,
                display_requested,
            )
        except FileNotFoundError:
            abort(404)

        display_limit = len(visible_results)
        recent = recent_queries()
        display_options = available_display_limits(total_results)
        selected_display_value, custom_display_limit = _display_selection_state(
            display_limit,
            display_options,
        )
        return render_template(
            "search.html",
            current_query=record["query_text"],
            query_record=record,
            results=visible_results.to_dict(orient="records"),
            total_results=total_results,
            display_limit=display_limit,
            display_options=display_options,
            selected_display_value=selected_display_value,
            custom_display_limit=custom_display_limit,
            recent_queries=recent,
            created_state=request.args.get("state"),
            error_message="",
        )

    @app.get("/api/query-suggestions")
    def query_suggestions():
        prefix = request.args.get("q", "")
        limit_raw = request.args.get("limit", "8")
        try:
            limit = max(0, min(int(limit_raw), 20))
        except ValueError:
            limit = 8
        return jsonify({"suggestions": autocomplete_queries(prefix, limit=limit)})

    @app.get("/queries/<query_id>/download")
    def download_query_results(query_id: str):
        try:
            record = get_query_record(query_id)
        except FileNotFoundError:
            abort(404)

        result_file = ranking_backend.resolve_result_path(record["result_file"])
        if not result_file.exists():
            abort(404)
        return send_file(result_file, as_attachment=True)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
