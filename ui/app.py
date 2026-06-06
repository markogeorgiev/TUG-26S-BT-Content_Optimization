from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

from retriever import run_hybrid_rankings as ranking_backend
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
from ui.services.feature_ui_service import (
    DEFAULT_RANK_FEATURE_PLOT_KIND,
    content_feature_document_count,
    get_article_features,
    get_or_create_per_model_rank_feature_plots,
    get_or_create_rank_feature_plots,
    rank_feature_plot_kind_options,
    resolve_article,
    resolve_per_model_rank_feature_plot_path,
    resolve_rank_feature_plot_path,
    saved_ranking_query_options,
    search_articles,
)
from ui.services.ranking_ui_service import (
    DEFAULT_DISPLAY_LIMIT,
    available_display_limits,
    autocomplete_queries,
    execute_or_load_query,
    get_per_model_config,
    get_query_record,
    load_per_model_query_results,
    load_query_results,
    recent_queries,
)
from ui.services.rank_relationship_service import (
    get_or_create_per_model_rank_relationship_analysis,
    get_or_create_rank_relationship_analysis,
    get_per_model_rank_relationship_metadata,
    get_rank_relationship_metadata,
    rank_relationship_query_records,
    resolve_per_model_rank_relationship_output,
    resolve_rank_relationship_output,
)


WORKSPACE_CONFIGS: dict[str, dict[str, Any]] = {
    "hybrid": {
        "label": "Hybrid",
        "nav_note": "BM25 + SBERT + PageRank",
        "header_note": "Weighted document retrieval with BM25 0.4, SBERT 0.4, and PageRank 0.2.",
        "query_heading": "Run Or Reuse A Hybrid Ranking",
        "query_note": (
            "New queries are executed once with the weighted hybrid retriever and then reused "
            "from the saved parquet file."
        ),
        "query_button_label": "Run hybrid query",
        "ranking_basis_title": "Weighted hybrid ranking",
        "ranking_basis_summary": (
            "Ranks documents with the stored hybrid score built from lexical matching, "
            "semantic similarity, and graph authority."
        ),
        "results_partial": "partials/results_table.html",
        "rank_relationship_heading": "Advanced Hybrid Rank Relationship Analysis",
        "rank_relationship_note": (
            "Uses all saved full-ranking queries, averages hybrid rank and hybrid score per "
            "article, then analyzes how content features relate to average hybrid rank."
        ),
        "rank_feature_heading": "Hybrid Rank Feature Plots",
        "rank_feature_note": (
            "Select one or more saved hybrid rankings and analyze how article features behave "
            "across their top 100 positions."
        ),
        "experiments_heading": "Hybrid Content Change Experiments",
        "experiments_note": (
            "Edit one article against a saved query, simulate link changes, and compare the "
            "temporary hybrid rerank against the stored baseline."
        ),
        "model_key": None,
        "experiment_model_key": "hybrid",
        "results_page_title": "Hybrid ranking preview",
    },
    "dense": {
        "label": "Dense",
        "nav_note": "SBERT semantic retrieval",
        "header_note": "Semantic document retrieval driven by SBERT chunk evidence aggregated back to full articles.",
        "query_heading": "Run Or Reuse A Dense Ranking",
        "query_note": (
            "The saved hybrid query file is reused, then reordered by SBERT-only semantic "
            "scores so the dense view stays aligned with the same stored query history."
        ),
        "query_button_label": "View dense ranking",
        "ranking_basis_title": "SBERT semantic ranking",
        "ranking_basis_summary": (
            "Reorders the saved query result by SBERT-only semantic similarity while keeping "
            "the hybrid and component columns visible for comparison."
        ),
        "results_partial": "partials/per_model_results_table.html",
        "rank_relationship_heading": "Advanced Dense Rank Relationship Analysis",
        "rank_relationship_note": (
            "Builds the target rank from SBERT-only ordering across the saved full-ranking "
            "queries, then analyzes how content features relate to dense retrieval behavior."
        ),
        "rank_feature_heading": "Dense Rank Feature Plots",
        "rank_feature_note": (
            "Select saved queries and study how article features behave across SBERT-only "
            "ranking positions in the top 100."
        ),
        "experiments_heading": "Dense Content Change Experiments",
        "experiments_note": (
            "Edit one article against a saved query, recompute temporary lexical, semantic, "
            "and local PageRank effects, then inspect the SBERT-driven rerank."
        ),
        "model_key": "sbert",
        "experiment_model_key": "sbert",
        "results_page_title": "Dense ranking preview",
    },
    "sparse": {
        "label": "Sparse",
        "nav_note": "BM25 lexical retrieval",
        "header_note": "Lexical document retrieval driven by BM25 token overlap and IDF-weighted matching.",
        "query_heading": "Run Or Reuse A Sparse Ranking",
        "query_note": (
            "The saved hybrid query file is reused, then reordered by BM25-only lexical scores "
            "so the sparse view stays aligned with the same stored query history."
        ),
        "query_button_label": "View sparse ranking",
        "ranking_basis_title": "BM25 lexical ranking",
        "ranking_basis_summary": (
            "Reorders the saved query result by BM25-only lexical matching while keeping the "
            "hybrid and component columns visible for comparison."
        ),
        "results_partial": "partials/per_model_results_table.html",
        "rank_relationship_heading": "Advanced Sparse Rank Relationship Analysis",
        "rank_relationship_note": (
            "Builds the target rank from BM25-only ordering across the saved full-ranking "
            "queries, then analyzes how content features relate to sparse retrieval behavior."
        ),
        "rank_feature_heading": "Sparse Rank Feature Plots",
        "rank_feature_note": (
            "Select saved queries and study how article features behave across BM25-only "
            "ranking positions in the top 100."
        ),
        "experiments_heading": "Sparse Content Change Experiments",
        "experiments_note": (
            "Edit one article against a saved query, recompute temporary lexical, semantic, "
            "and local PageRank effects, then inspect the BM25-driven rerank."
        ),
        "model_key": "bm25",
        "experiment_model_key": "bm25",
        "results_page_title": "Sparse ranking preview",
    },
}

WORKSPACE_ORDER = ["hybrid", "dense", "sparse"]

SECTION_ITEMS = [
    {"key": "query", "label": "Query", "endpoint": "workspace_query"},
    {"key": "rank_feature_plots", "label": "Rank Feature Plots", "endpoint": "workspace_rank_feature_plots"},
    {
        "key": "rank_relationship",
        "label": "Advanced Rank Relationship Analysis",
        "endpoint": "workspace_rank_relationship",
    },
    {
        "key": "experiments",
        "label": "Content Change Experiments",
        "endpoint": "workspace_content_change_experiments",
    },
]


def normalize_workspace(workspace: str | None) -> str:
    normalized = str(workspace or "").strip().lower()
    if normalized not in WORKSPACE_CONFIGS:
        raise ValueError(f"Unknown workspace: {workspace}")
    return normalized


def workspace_config(workspace: str | None) -> dict[str, Any]:
    return WORKSPACE_CONFIGS[normalize_workspace(workspace)]


def workspace_model_key(workspace: str | None) -> str | None:
    return str(workspace_config(workspace).get("model_key") or "") or None


def workspace_experiment_model_key(workspace: str | None) -> str:
    return str(workspace_config(workspace)["experiment_model_key"])


def legacy_workspace_from_model(model_key: str | None) -> str:
    normalized = str(model_key or "").strip().lower()
    if normalized == "sbert":
        return "dense"
    if normalized == "bm25":
        return "sparse"
    return "hybrid"


def build_primary_nav(current_workspace: str | None, current_section: str | None) -> list[dict[str, Any]]:
    active_workspace = str(current_workspace or "").strip().lower()
    active_section = str(current_section or "").strip().lower()
    items: list[dict[str, Any]] = []

    for workspace in WORKSPACE_ORDER:
        config = WORKSPACE_CONFIGS[workspace]
        items.append(
            {
                "label": config["label"],
                "note": config["nav_note"],
                "url": url_for("workspace_query", workspace=workspace),
                "is_active": active_workspace == workspace,
            }
        )

    items.append(
        {
            "label": "Content Features",
            "note": "Corpus-wide article diagnostics",
            "url": url_for("article_features"),
            "is_active": active_section == "content_features",
        }
    )
    return items


def build_workspace_section_nav(current_workspace: str | None, current_section: str | None) -> list[dict[str, Any]]:
    if not current_workspace:
        return []

    workspace = normalize_workspace(current_workspace)
    active_section = str(current_section or "").strip().lower()
    items: list[dict[str, Any]] = []
    for section in SECTION_ITEMS:
        items.append(
            {
                "label": section["label"],
                "url": url_for(section["endpoint"], workspace=workspace),
                "is_active": active_section == section["key"],
            }
        )
    return items


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

    @app.context_processor
    def inject_shell_navigation():
        return {
            "build_primary_nav": build_primary_nav,
            "build_workspace_section_nav": build_workspace_section_nav,
        }

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
            workspace_key="",
            section_key="content_features",
            header_note="Article-level corpus diagnostics independent of ranking model.",
            article=article,
            article_search=article_search,
            feature_document_count=feature_document_count,
            error_message=error_message,
        )

    def render_workspace_query_page(
        workspace: str,
        *,
        current_query: str = "",
        query_record: dict | None = None,
        results: list[dict] | None = None,
        total_results: int = 0,
        display_limit: int = DEFAULT_DISPLAY_LIMIT,
        created_state: str | None = None,
        error_message: str = "",
    ):
        workspace = normalize_workspace(workspace)
        config = WORKSPACE_CONFIGS[workspace]
        model_key = workspace_model_key(workspace)
        model_info = get_per_model_config(model_key) if model_key else None
        display_options = available_display_limits(total_results)
        selected_display_value, custom_display_limit = _display_selection_state(
            display_limit,
            display_options,
        )

        return render_template(
            "workspace_query.html",
            workspace_key=workspace,
            section_key="query",
            header_note=config["header_note"],
            workspace_config=config,
            model_info=model_info,
            current_query=current_query,
            query_record=query_record,
            results=results or [],
            total_results=total_results,
            display_limit=display_limit,
            display_options=display_options,
            selected_display_value=selected_display_value,
            custom_display_limit=custom_display_limit,
            recent_queries=recent_queries(),
            created_state=created_state,
            error_message=error_message,
            query_form_action=url_for("workspace_search", workspace=workspace),
            query_results_endpoint="workspace_query_results",
            workspace_results_partial=config["results_partial"],
        )

    def render_workspace_rank_relationship_page(
        workspace: str,
        *,
        analysis: dict | None = None,
        error_message: str = "",
    ):
        workspace = normalize_workspace(workspace)
        config = WORKSPACE_CONFIGS[workspace]
        is_hybrid = workspace == "hybrid"
        model_key = workspace_model_key(workspace)

        query_options = rank_relationship_query_records()
        usable_query_count = sum(
            1
            for query in query_options
            if query["has_result_file"] and query["is_full_ranking"]
        )

        if analysis:
            analysis["summary_url"] = url_for(
                "workspace_rank_relationship_output",
                workspace=workspace,
                collection_key=analysis["collection_key"],
                output_name="summary",
            )
            analysis["dot_plot_url"] = url_for(
                "workspace_rank_relationship_output",
                workspace=workspace,
                collection_key=analysis["collection_key"],
                output_name="dot_plot",
            )
            analysis["diverging_bar_url"] = url_for(
                "workspace_rank_relationship_output",
                workspace=workspace,
                collection_key=analysis["collection_key"],
                output_name="diverging_bar",
            )

            if is_hybrid:
                analysis["model_correlations_url"] = url_for(
                    "workspace_rank_relationship_output",
                    workspace=workspace,
                    collection_key=analysis["collection_key"],
                    output_name="model_correlations",
                )
                analysis["model_importance_url"] = url_for(
                    "workspace_rank_relationship_output",
                    workspace=workspace,
                    collection_key=analysis["collection_key"],
                    output_name="model_importance",
                )
                analysis["model_correlation_heatmaps_url"] = url_for(
                    "workspace_rank_relationship_output",
                    workspace=workspace,
                    collection_key=analysis["collection_key"],
                    output_name="model_correlation_heatmaps",
                )
                analysis["model_importance_heatmaps_url"] = url_for(
                    "workspace_rank_relationship_output",
                    workspace=workspace,
                    collection_key=analysis["collection_key"],
                    output_name="model_importance_heatmaps",
                )
            else:
                analysis["bm25_sbert_model_correlations_url"] = url_for(
                    "workspace_rank_relationship_output",
                    workspace=workspace,
                    collection_key=analysis["collection_key"],
                    output_name="bm25_sbert_model_correlations",
                )
                analysis["bm25_sbert_model_correlation_heatmaps_url"] = url_for(
                    "workspace_rank_relationship_output",
                    workspace=workspace,
                    collection_key=analysis["collection_key"],
                    output_name="bm25_sbert_model_correlation_heatmaps",
                )

        return render_template(
            "workspace_rank_relationship.html",
            workspace_key=workspace,
            section_key="rank_relationship",
            header_note=config["header_note"],
            workspace_config=config,
            is_hybrid=is_hybrid,
            model_info=get_per_model_config(model_key) if model_key else None,
            analysis=analysis,
            query_options=query_options,
            usable_query_count=usable_query_count,
            error_message=error_message,
            rank_relationship_run_action=url_for("workspace_run_rank_relationship", workspace=workspace),
        )

    def render_workspace_rank_feature_plots_page(
        workspace: str,
        *,
        selected_query_ids: list[str] | None = None,
        selected_plot_kind: str = DEFAULT_RANK_FEATURE_PLOT_KIND,
        analysis: dict | None = None,
        error_message: str = "",
    ):
        workspace = normalize_workspace(workspace)
        config = WORKSPACE_CONFIGS[workspace]
        is_hybrid = workspace == "hybrid"
        model_key = workspace_model_key(workspace)

        query_options = saved_ranking_query_options()
        plot_kind_options = rank_feature_plot_kind_options()
        selected_ids = set(selected_query_ids or [])

        if analysis:
            for plot_file in analysis.get("plot_files", []):
                plot_file["url"] = url_for(
                    "workspace_rank_feature_plot_image",
                    workspace=workspace,
                    collection_key=analysis["collection_key"],
                    filename=plot_file["file_name"],
                )

        return render_template(
            "workspace_rank_feature_plots.html",
            workspace_key=workspace,
            section_key="rank_feature_plots",
            header_note=config["header_note"],
            workspace_config=config,
            is_hybrid=is_hybrid,
            model_info=get_per_model_config(model_key) if model_key else None,
            query_options=query_options,
            plot_kind_options=plot_kind_options,
            selected_query_ids=selected_ids,
            selected_plot_kind=selected_plot_kind,
            analysis=analysis,
            error_message=error_message,
            rank_plot_form_action=url_for("workspace_rank_feature_plots", workspace=workspace),
        )

    def render_content_change_experiments_page(
        workspace: str,
        *,
        selected_query_id: str = "",
        article: dict | None = None,
        article_search: str = "",
        baseline_entry: dict | None = None,
        result: dict | None = None,
        form_values: dict | None = None,
        error_message: str = "",
    ):
        workspace = normalize_workspace(workspace)
        config = WORKSPACE_CONFIGS[workspace]
        experiment_model_key = workspace_experiment_model_key(workspace)
        model_info = get_per_model_config(experiment_model_key) if workspace != "hybrid" else None

        if baseline_entry is None and article and selected_query_id:
            try:
                baseline_entry = get_article_baseline_ranking_entry(
                    selected_query_id,
                    int(article["page_id"]),
                    model_key=experiment_model_key,
                )
            except (FileNotFoundError, ValueError) as exc:
                error_message = error_message or str(exc)

        return render_template(
            "content_change_experiments.html",
            workspace_key=workspace,
            section_key="experiments",
            header_note=config["header_note"],
            workspace_config=config,
            model_info=model_info,
            query_options=saved_experiment_query_options(),
            selected_query_id=selected_query_id,
            article=article,
            article_search=article_search,
            baseline_entry=baseline_entry,
            result=result,
            form_values=form_values or {},
            error_message=error_message,
            select_form_action=url_for(
                "workspace_select_content_change_experiment_article",
                workspace=workspace,
            ),
            run_form_action=url_for(
                "workspace_run_content_change_experiment",
                workspace=workspace,
            ),
        )

    @app.get("/")
    def root():
        return redirect(url_for("workspace_query", workspace="hybrid"))

    @app.get("/<workspace>/query")
    def workspace_query(workspace: str):
        workspace = normalize_workspace(workspace)
        return render_workspace_query_page(
            workspace,
            error_message=request.args.get("error", ""),
        )

    @app.post("/<workspace>/query")
    def workspace_search(workspace: str):
        workspace = normalize_workspace(workspace)
        query = request.form.get("query", "").strip()
        if not query:
            return redirect(
                url_for(
                    "workspace_query",
                    workspace=workspace,
                    error="Please enter a query.",
                )
            )

        display_limit_raw = _resolve_display_limit_from_request("display_limit")
        record, _, created = execute_or_load_query(query)
        return redirect(
            url_for(
                "workspace_query_results",
                workspace=workspace,
                query_id=record["query_id"],
                display=display_limit_raw,
                state="created" if created else "reused",
            )
        )

    @app.get("/<workspace>/queries/<query_id>")
    def workspace_query_results(workspace: str, query_id: str):
        workspace = normalize_workspace(workspace)
        display_requested = request.args.get("display", str(DEFAULT_DISPLAY_LIMIT)).strip()

        try:
            if workspace == "hybrid":
                record, visible_results, total_results = load_query_results(
                    query_id,
                    display_requested,
                )
            else:
                record, visible_results, total_results, _ = load_per_model_query_results(
                    query_id,
                    workspace_model_key(workspace),
                    display_requested,
                )
        except FileNotFoundError:
            abort(404)
        except ValueError as exc:
            return render_workspace_query_page(
                workspace,
                error_message=str(exc),
            )

        return render_workspace_query_page(
            workspace,
            current_query=record["query_text"],
            query_record=record,
            results=visible_results.to_dict(orient="records"),
            total_results=total_results,
            display_limit=len(visible_results),
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

    @app.get("/<workspace>/rank-relationship")
    def workspace_rank_relationship(workspace: str):
        workspace = normalize_workspace(workspace)
        return render_workspace_rank_relationship_page(
            workspace,
            error_message=request.args.get("error", ""),
        )

    @app.post("/<workspace>/rank-relationship/run")
    def workspace_run_rank_relationship(workspace: str):
        workspace = normalize_workspace(workspace)
        try:
            force = request.form.get("force") == "1"
            if workspace == "hybrid":
                analysis = get_or_create_rank_relationship_analysis(force=force)
            else:
                analysis = get_or_create_per_model_rank_relationship_analysis(
                    workspace_model_key(workspace),
                    force=force,
                )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return render_workspace_rank_relationship_page(
                workspace,
                error_message=str(exc),
            )

        return redirect(
            url_for(
                "workspace_rank_relationship_results",
                workspace=workspace,
                collection_key=analysis["collection_key"],
            )
        )

    @app.get("/<workspace>/rank-relationship/<collection_key>")
    def workspace_rank_relationship_results(workspace: str, collection_key: str):
        workspace = normalize_workspace(workspace)
        try:
            if workspace == "hybrid":
                analysis = get_rank_relationship_metadata(collection_key)
            else:
                analysis = get_per_model_rank_relationship_metadata(collection_key)
        except FileNotFoundError:
            abort(404)

        return render_workspace_rank_relationship_page(
            workspace,
            analysis=analysis,
        )

    @app.get("/<workspace>/rank-relationship/<collection_key>/<output_name>")
    def workspace_rank_relationship_output(
        workspace: str,
        collection_key: str,
        output_name: str,
    ):
        workspace = normalize_workspace(workspace)
        try:
            if workspace == "hybrid":
                output_path = resolve_rank_relationship_output(collection_key, output_name)
            else:
                output_path = resolve_per_model_rank_relationship_output(collection_key, output_name)
        except FileNotFoundError:
            abort(404)

        if output_path.suffix.lower() == ".png":
            return send_file(output_path, mimetype="image/png")
        return send_file(output_path, as_attachment=True, mimetype="text/csv")

    @app.get("/<workspace>/rank-feature-plots")
    def workspace_rank_feature_plots(workspace: str):
        workspace = normalize_workspace(workspace)
        selected_query_ids = request.args.getlist("query_id")
        selected_plot_kind = request.args.get(
            "plot_kind",
            DEFAULT_RANK_FEATURE_PLOT_KIND,
        ).strip()

        if not selected_query_ids:
            return render_workspace_rank_feature_plots_page(
                workspace,
                selected_plot_kind=selected_plot_kind,
            )

        try:
            if workspace == "hybrid":
                analysis = get_or_create_rank_feature_plots(
                    selected_query_ids,
                    plot_kind=selected_plot_kind,
                )
            else:
                analysis = get_or_create_per_model_rank_feature_plots(
                    selected_query_ids,
                    model_key=workspace_model_key(workspace),
                    plot_kind=selected_plot_kind,
                )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return render_workspace_rank_feature_plots_page(
                workspace,
                selected_query_ids=selected_query_ids,
                selected_plot_kind=selected_plot_kind,
                error_message=str(exc),
            )

        return render_workspace_rank_feature_plots_page(
            workspace,
            selected_query_ids=analysis["query_ids"],
            selected_plot_kind=analysis["plot_kind"],
            analysis=analysis,
        )

    @app.get("/<workspace>/rank-feature-plots/<collection_key>/<path:filename>")
    def workspace_rank_feature_plot_image(workspace: str, collection_key: str, filename: str):
        workspace = normalize_workspace(workspace)
        try:
            if workspace == "hybrid":
                plot_path = resolve_rank_feature_plot_path(collection_key, filename)
            else:
                plot_path = resolve_per_model_rank_feature_plot_path(collection_key, filename)
        except FileNotFoundError:
            abort(404)
        return send_file(plot_path, mimetype="image/png")

    @app.get("/<workspace>/content-change-experiments")
    def workspace_content_change_experiments(workspace: str):
        workspace = normalize_workspace(workspace)
        query_id = request.args.get("query_id", "").strip()
        page_id = request.args.get("page_id", "").strip()
        if not query_id or not page_id:
            return render_content_change_experiments_page(
                workspace,
                selected_query_id=query_id,
                error_message=request.args.get("error", ""),
            )

        try:
            get_experiment_query_record(query_id)
            article = get_experiment_article(int(page_id))
        except (FileNotFoundError, ValueError) as exc:
            return render_content_change_experiments_page(
                workspace,
                selected_query_id=query_id,
                error_message=str(exc),
            )

        return render_content_change_experiments_page(
            workspace,
            selected_query_id=query_id,
            article=article,
            article_search=str(article["title"]),
        )

    @app.post("/<workspace>/content-change-experiments/select")
    def workspace_select_content_change_experiment_article(workspace: str):
        workspace = normalize_workspace(workspace)
        query_id = request.form.get("query_id", "").strip()
        page_id = request.form.get("page_id", "").strip()
        article_search = request.form.get("article", "").strip()

        if not query_id:
            return redirect(
                url_for(
                    "workspace_content_change_experiments",
                    workspace=workspace,
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
                workspace,
                selected_query_id=query_id,
                article_search=article_search,
                error_message=str(exc),
            )

        return redirect(
            url_for(
                "workspace_content_change_experiments",
                workspace=workspace,
                query_id=query_id,
                page_id=article["page_id"],
            )
        )

    @app.post("/<workspace>/content-change-experiments/run")
    def workspace_run_content_change_experiment(workspace: str):
        workspace = normalize_workspace(workspace)
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
                model_key=workspace_experiment_model_key(workspace),
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            article = None
            if page_id_raw.isdigit():
                try:
                    article = get_experiment_article(int(page_id_raw))
                except (FileNotFoundError, ValueError):
                    article = None
            return render_content_change_experiments_page(
                workspace,
                selected_query_id=query_id,
                article=article,
                article_search=str(article["title"]) if article else "",
                form_values=form_values,
                error_message=str(exc),
            )

        return render_content_change_experiments_page(
            workspace,
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
        return redirect(url_for("article_features"))

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

    @app.get("/search")
    def legacy_search_index():
        return redirect(url_for("workspace_query", workspace="hybrid"))

    @app.post("/search")
    def legacy_search_submit():
        return workspace_search("hybrid")

    @app.get("/queries/<query_id>")
    def legacy_query_results(query_id: str):
        redirect_args: dict[str, Any] = {
            "workspace": "hybrid",
            "query_id": query_id,
        }
        display = request.args.get("display", "").strip()
        state = request.args.get("state", "").strip()
        if display:
            redirect_args["display"] = display
        if state:
            redirect_args["state"] = state
        return redirect(url_for("workspace_query_results", **redirect_args))

    @app.get("/rank-relationship")
    def legacy_rank_relationship():
        return redirect(url_for("workspace_rank_relationship", workspace="hybrid"))

    @app.post("/rank-relationship/run")
    def legacy_run_rank_relationship():
        return workspace_run_rank_relationship("hybrid")

    @app.get("/rank-relationship/<collection_key>")
    def legacy_rank_relationship_results(collection_key: str):
        return redirect(
            url_for(
                "workspace_rank_relationship_results",
                workspace="hybrid",
                collection_key=collection_key,
            )
        )

    @app.get("/rank-relationship/<collection_key>/<output_name>")
    def legacy_rank_relationship_output(collection_key: str, output_name: str):
        return redirect(
            url_for(
                "workspace_rank_relationship_output",
                workspace="hybrid",
                collection_key=collection_key,
                output_name=output_name,
            )
        )

    @app.get("/per-model-ranking")
    def legacy_per_model_ranking():
        return redirect(
            url_for(
                "workspace_query",
                workspace=legacy_workspace_from_model(request.args.get("model")),
            )
        )

    @app.post("/per-model-ranking/search")
    def legacy_per_model_search():
        return workspace_search(legacy_workspace_from_model(request.form.get("model")))

    @app.get("/per-model-ranking/queries/<query_id>")
    def legacy_per_model_query_results(query_id: str):
        redirect_args: dict[str, Any] = {
            "workspace": legacy_workspace_from_model(request.args.get("model")),
            "query_id": query_id,
        }
        display = request.args.get("display", "").strip()
        state = request.args.get("state", "").strip()
        if display:
            redirect_args["display"] = display
        if state:
            redirect_args["state"] = state
        return redirect(url_for("workspace_query_results", **redirect_args))

    @app.get("/per-model-rank-relationship")
    def legacy_per_model_rank_relationship():
        return redirect(
            url_for(
                "workspace_rank_relationship",
                workspace=legacy_workspace_from_model(request.args.get("model")),
            )
        )

    @app.post("/per-model-rank-relationship/run")
    def legacy_run_per_model_rank_relationship():
        return workspace_run_rank_relationship(legacy_workspace_from_model(request.form.get("model")))

    @app.get("/per-model-rank-relationship/<collection_key>")
    def legacy_per_model_rank_relationship_results(collection_key: str):
        workspace = "dense" if collection_key.startswith("sbert_") else "sparse"
        return redirect(
            url_for(
                "workspace_rank_relationship_results",
                workspace=workspace,
                collection_key=collection_key,
            )
        )

    @app.get("/per-model-rank-relationship/<collection_key>/<output_name>")
    def legacy_per_model_rank_relationship_output(collection_key: str, output_name: str):
        workspace = "dense" if collection_key.startswith("sbert_") else "sparse"
        return redirect(
            url_for(
                "workspace_rank_relationship_output",
                workspace=workspace,
                collection_key=collection_key,
                output_name=output_name,
            )
        )

    @app.get("/content-change-experiments")
    def legacy_content_change_experiments():
        redirect_args: dict[str, Any] = {"workspace": "hybrid"}
        query_id = request.args.get("query_id", "").strip()
        page_id = request.args.get("page_id", "").strip()
        error = request.args.get("error", "").strip()
        if query_id:
            redirect_args["query_id"] = query_id
        if page_id:
            redirect_args["page_id"] = page_id
        if error:
            redirect_args["error"] = error
        return redirect(url_for("workspace_content_change_experiments", **redirect_args))

    @app.post("/content-change-experiments/select")
    def legacy_select_content_change_experiment_article():
        return workspace_select_content_change_experiment_article("hybrid")

    @app.post("/content-change-experiments/run")
    def legacy_run_content_change_experiment():
        return workspace_run_content_change_experiment("hybrid")

    @app.get("/features/rank-plots")
    def legacy_rank_feature_plots():
        return redirect(
            url_for(
                "workspace_rank_feature_plots",
                workspace="hybrid",
                query_id=request.args.getlist("query_id"),
                plot_kind=request.args.get("plot_kind", DEFAULT_RANK_FEATURE_PLOT_KIND),
            )
        )

    @app.get("/features/rank-plots/<collection_key>/<path:filename>")
    def legacy_rank_feature_plot_image(collection_key: str, filename: str):
        return redirect(
            url_for(
                "workspace_rank_feature_plot_image",
                workspace="hybrid",
                collection_key=collection_key,
                filename=filename,
            )
        )

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
