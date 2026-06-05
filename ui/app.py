from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

from retriever import run_hybrid_rankings as ranking_backend
from ui.services.feature_ui_service import (
    DEFAULT_RANK_FEATURE_PLOT_KIND,
    content_feature_document_count,
    get_article_features,
    get_or_create_rank_feature_plots,
    rank_feature_plot_kind_options,
    resolve_article,
    resolve_rank_feature_plot_path,
    saved_ranking_query_options,
    search_articles,
)
from ui.services.content_experiment_service import (
    build_link_edit_request,
    get_article_baseline_ranking_entry,
    get_experiment_article,
    get_experiment_query_record,
    resolve_experiment_article,
    run_content_change_experiment as run_content_change_experiment_backend,
    saved_experiment_query_options,
    search_experiment_articles,
)
from ui.services.ranking_ui_service import (
    DEFAULT_DISPLAY_LIMIT,
    available_display_limits,
    autocomplete_queries,
    execute_or_load_query,
    get_query_record,
    load_query_results,
    recent_queries,
)
from ui.services.rank_relationship_service import (
    get_or_create_rank_relationship_analysis,
    get_rank_relationship_metadata,
    rank_relationship_query_records,
    resolve_rank_relationship_output,
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

    def render_article_features_page(
        *,
        article: dict | None = None,
        article_search: str = "",
        error_message: str = "",
    ):
        try:
            feature_document_count = content_feature_document_count()
        except (FileNotFoundError, ValueError) as exc:
            feature_document_count = 0
            error_message = error_message or str(exc)

        return render_template(
            "features_article.html",
            article=article,
            article_search=article_search,
            feature_document_count=feature_document_count,
            error_message=error_message,
        )

    def render_rank_feature_plots_page(
        *,
        selected_query_ids: list[str] | None = None,
        selected_plot_kind: str = DEFAULT_RANK_FEATURE_PLOT_KIND,
        analysis: dict | None = None,
        error_message: str = "",
    ):
        query_options = saved_ranking_query_options()
        plot_kind_options = rank_feature_plot_kind_options()
        selected_ids = set(selected_query_ids or [])
        if analysis:
            for plot_file in analysis.get("plot_files", []):
                plot_file["url"] = url_for(
                    "rank_feature_plot_image",
                    collection_key=analysis["collection_key"],
                    filename=plot_file["file_name"],
                )

        return render_template(
            "features_rank_plots.html",
            query_options=query_options,
            plot_kind_options=plot_kind_options,
            selected_query_ids=selected_ids,
            selected_plot_kind=selected_plot_kind,
            analysis=analysis,
            error_message=error_message,
        )

    def render_rank_relationship_page(
        *,
        analysis: dict | None = None,
        error_message: str = "",
    ):
        query_options = rank_relationship_query_records()
        usable_query_count = sum(
            1
            for query in query_options
            if query["has_result_file"] and query["is_full_ranking"]
        )
        if analysis:
            analysis["summary_url"] = url_for(
                "rank_relationship_output",
                collection_key=analysis["collection_key"],
                output_name="summary",
            )
            analysis["dot_plot_url"] = url_for(
                "rank_relationship_output",
                collection_key=analysis["collection_key"],
                output_name="dot_plot",
            )
            analysis["diverging_bar_url"] = url_for(
                "rank_relationship_output",
                collection_key=analysis["collection_key"],
                output_name="diverging_bar",
            )
            analysis["model_correlations_url"] = url_for(
                "rank_relationship_output",
                collection_key=analysis["collection_key"],
                output_name="model_correlations",
            )
            analysis["model_importance_url"] = url_for(
                "rank_relationship_output",
                collection_key=analysis["collection_key"],
                output_name="model_importance",
            )
            analysis["model_correlation_heatmaps_url"] = url_for(
                "rank_relationship_output",
                collection_key=analysis["collection_key"],
                output_name="model_correlation_heatmaps",
            )
            analysis["model_importance_heatmaps_url"] = url_for(
                "rank_relationship_output",
                collection_key=analysis["collection_key"],
                output_name="model_importance_heatmaps",
            )

        return render_template(
            "rank_relationship.html",
            analysis=analysis,
            query_options=query_options,
            usable_query_count=usable_query_count,
            error_message=error_message,
        )

    def render_content_change_experiments_page(
        *,
        selected_query_id: str = "",
        article: dict | None = None,
        article_search: str = "",
        baseline_entry: dict | None = None,
        result: dict | None = None,
        form_values: dict | None = None,
        error_message: str = "",
    ):
        if baseline_entry is None and article and selected_query_id:
            try:
                baseline_entry = get_article_baseline_ranking_entry(
                    selected_query_id,
                    int(article["page_id"]),
                )
            except (FileNotFoundError, ValueError) as exc:
                error_message = error_message or str(exc)

        return render_template(
            "content_change_experiments.html",
            query_options=saved_experiment_query_options(),
            selected_query_id=selected_query_id,
            article=article,
            article_search=article_search,
            baseline_entry=baseline_entry,
            result=result,
            form_values=form_values or {},
            error_message=error_message,
        )

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

    @app.get("/rank-relationship")
    def rank_relationship():
        return render_rank_relationship_page()

    @app.post("/rank-relationship/run")
    def run_rank_relationship():
        force = request.form.get("force") == "1"
        try:
            analysis = get_or_create_rank_relationship_analysis(force=force)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return render_rank_relationship_page(error_message=str(exc))

        return redirect(
            url_for(
                "rank_relationship_results",
                collection_key=analysis["collection_key"],
            )
        )

    @app.get("/rank-relationship/<collection_key>")
    def rank_relationship_results(collection_key: str):
        try:
            analysis = get_rank_relationship_metadata(collection_key)
        except FileNotFoundError:
            abort(404)
        return render_rank_relationship_page(analysis=analysis)

    @app.get("/rank-relationship/<collection_key>/<output_name>")
    def rank_relationship_output(collection_key: str, output_name: str):
        try:
            output_path = resolve_rank_relationship_output(collection_key, output_name)
        except FileNotFoundError:
            abort(404)
        if output_path.suffix.lower() == ".png":
            return send_file(output_path, mimetype="image/png")
        return send_file(output_path, as_attachment=True, mimetype="text/csv")

    @app.get("/content-change-experiments")
    def content_change_experiments():
        query_id = request.args.get("query_id", "").strip()
        page_id = request.args.get("page_id", "").strip()
        if not query_id or not page_id:
            return render_content_change_experiments_page(
                selected_query_id=query_id,
                error_message=request.args.get("error", ""),
            )

        try:
            get_experiment_query_record(query_id)
            article = get_experiment_article(int(page_id))
        except (FileNotFoundError, ValueError) as exc:
            return render_content_change_experiments_page(
                selected_query_id=query_id,
                error_message=str(exc),
            )

        return render_content_change_experiments_page(
            selected_query_id=query_id,
            article=article,
            article_search=str(article["title"]),
        )

    @app.post("/content-change-experiments/select")
    def select_content_change_experiment_article():
        query_id = request.form.get("query_id", "").strip()
        page_id = request.form.get("page_id", "").strip()
        article_search = request.form.get("article", "").strip()

        if not query_id:
            return redirect(
                url_for(
                    "content_change_experiments",
                    error="Please select a saved query.",
                )
            )

        try:
            get_experiment_query_record(query_id)
            if page_id:
                article = get_experiment_article(int(page_id))
            else:
                article = resolve_experiment_article(article_search)
                if article is None:
                    raise ValueError(f'No article matched "{article_search}".')
        except (FileNotFoundError, ValueError) as exc:
            return render_content_change_experiments_page(
                selected_query_id=query_id,
                article_search=article_search,
                error_message=str(exc),
            )

        return redirect(
            url_for(
                "content_change_experiments",
                query_id=query_id,
                page_id=article["page_id"],
            )
        )

    @app.post("/content-change-experiments/run")
    def run_content_change_experiment():
        query_id = request.form.get("query_id", "").strip()
        page_id_raw = request.form.get("page_id", "").strip()
        edited_text = request.form.get("edited_text", "")
        form_values = {
            "edited_text": edited_text,
            "add_outgoing_page_ids": request.form.get("add_outgoing_page_ids", ""),
            "remove_outgoing_page_ids": request.form.get("remove_outgoing_page_ids", ""),
            "add_incoming_page_ids": request.form.get("add_incoming_page_ids", ""),
            "remove_incoming_page_ids": request.form.get("remove_incoming_page_ids", ""),
        }

        try:
            page_id = int(page_id_raw)
            article = get_experiment_article(page_id)
            link_edits = build_link_edit_request(request.form)
            result = run_content_change_experiment_backend(
                query_id=query_id,
                page_id=page_id,
                edited_text=edited_text,
                link_edits=link_edits,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            article = None
            if page_id_raw.isdigit():
                try:
                    article = get_experiment_article(int(page_id_raw))
                except (FileNotFoundError, ValueError):
                    article = None
            return render_content_change_experiments_page(
                selected_query_id=query_id,
                article=article,
                article_search=str(article["title"]) if article else "",
                form_values=form_values,
                error_message=str(exc),
            )

        return render_content_change_experiments_page(
            selected_query_id=query_id,
            article=article,
            article_search=str(article["title"]),
            result=result,
            form_values=form_values,
        )

    @app.get("/api/content-change-article-suggestions")
    def content_change_article_suggestions():
        search_text = request.args.get("q", "")
        limit_raw = request.args.get("limit", "10")
        try:
            limit = max(0, min(int(limit_raw), 20))
        except ValueError:
            limit = 10

        try:
            suggestions = search_experiment_articles(search_text, limit=limit)
        except (FileNotFoundError, ValueError):
            suggestions = []
        return jsonify({"suggestions": suggestions})

    @app.get("/features")
    def features():
        return redirect(url_for("rank_feature_plots"))

    @app.get("/features/articles")
    def article_features():
        article_search = request.args.get("article", "").strip()
        if not article_search:
            return render_article_features_page()

        try:
            article_match = resolve_article(article_search)
        except (FileNotFoundError, ValueError) as exc:
            return render_article_features_page(
                article_search=article_search,
                error_message=str(exc),
            )

        if article_match is None:
            return render_article_features_page(
                article_search=article_search,
                error_message=f'No article matched "{article_search}".',
            )
        return redirect(
            url_for(
                "article_feature_detail",
                page_id=article_match["page_id"],
            )
        )

    @app.get("/features/articles/<int:page_id>")
    def article_feature_detail(page_id: int):
        try:
            article = get_article_features(page_id)
        except FileNotFoundError:
            abort(404)
        except ValueError as exc:
            return render_article_features_page(error_message=str(exc))

        return render_article_features_page(
            article=article,
            article_search=str(article["title"]),
        )

    @app.get("/api/article-suggestions")
    def article_suggestions():
        search_text = request.args.get("q", "")
        limit_raw = request.args.get("limit", "10")
        try:
            limit = max(0, min(int(limit_raw), 20))
        except ValueError:
            limit = 10

        try:
            suggestions = search_articles(search_text, limit=limit)
        except (FileNotFoundError, ValueError):
            suggestions = []

        for suggestion in suggestions:
            suggestion["url"] = url_for(
                "article_feature_detail",
                page_id=suggestion["page_id"],
            )
        return jsonify({"suggestions": suggestions})

    @app.get("/features/rank-plots")
    def rank_feature_plots():
        selected_query_ids = request.args.getlist("query_id")
        selected_plot_kind = request.args.get(
            "plot_kind",
            DEFAULT_RANK_FEATURE_PLOT_KIND,
        ).strip()
        if not selected_query_ids:
            return render_rank_feature_plots_page(selected_plot_kind=selected_plot_kind)

        try:
            analysis = get_or_create_rank_feature_plots(
                selected_query_ids,
                plot_kind=selected_plot_kind,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return render_rank_feature_plots_page(
                selected_query_ids=selected_query_ids,
                selected_plot_kind=selected_plot_kind,
                error_message=str(exc),
            )

        return render_rank_feature_plots_page(
            selected_query_ids=analysis["query_ids"],
            selected_plot_kind=analysis["plot_kind"],
            analysis=analysis,
        )

    @app.get("/features/rank-plots/<collection_key>/<path:filename>")
    def rank_feature_plot_image(collection_key: str, filename: str):
        try:
            plot_path = resolve_rank_feature_plot_path(collection_key, filename)
        except FileNotFoundError:
            abort(404)
        return send_file(plot_path, mimetype="image/png")

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
