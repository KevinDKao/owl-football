import pandas as pd
import plotly.express as px
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import json


def register_callbacks(app):
    """
    Register all callback functions for the app
    """

    @app.callback(
        [
            Output("filtered-data", "data"),
            Output("results-info", "children"),
            Output("total-prospects", "children"),
            Output("total-drafted", "children"),
        ],
        [Input("apply-filters", "n_clicks"), Input("search-input", "value")],
        [
            State("position-filter", "value"),
            State("school-filter", "value"),
            State("sort-by", "value"),
            State("min-height", "value"),
            State("min-weight", "value"),
            State("min-forty", "value"),
            State("min-vertical", "value"),
        ],
    )
    def filter_data(
        n_clicks,
        search_term,
        position,
        school,
        sort_by,
        min_height,
        min_weight,
        min_forty,
        min_vertical,
    ):
        # Get the dataframe from the app's data store
        if not hasattr(app, 'df') or app.df is None:
            return [], "No data available", "0", "0"
            
        filtered_df = app.df.copy()
        
        if filtered_df.empty:
            return [], "No data available", "0", "0"

        # Apply position filter
        if position and position != "All":
            filtered_df = filtered_df[filtered_df["position_group"] == position]

        # Apply school filter
        if school and school != "All":
            filtered_df = filtered_df[filtered_df["school"] == school]

        # Apply min height filter if not "Any"
        if min_height and min_height != "Any":
            # Logic to filter by height would go here
            # This is a simplified approach since height is in a special format
            pass

        # Apply min weight filter
        if min_weight:
            filtered_df = filtered_df[filtered_df["weight_clean"] >= min_weight]

        # Apply 40 yard dash filter (lower is better)
        if min_forty:
            # Make sure the filter is clear - we want FASTER (lower) times
            # Only filter records with valid 40 yard dash times
            filtered_df = filtered_df[
                (
                    filtered_df["40_yard_dash"].notna()
                    & (filtered_df["40_yard_dash"] <= min_forty)
                )
                | (filtered_df["40_yard_dash"].isna())
            ]

        # Apply vertical jump filter (higher is better)
        if min_vertical:
            # Only filter records with valid vertical jump measurements
            filtered_df = filtered_df[
                (
                    filtered_df["vertical"].notna()
                    & (filtered_df["vertical"] >= min_vertical)
                )
                | (filtered_df["vertical"].isna())
            ]

        # Apply search term
        if search_term:
            filtered_df = filtered_df[
                filtered_df["name"].str.contains(search_term, case=False, na=False)
            ]

        # Filter to only show players predicted to be drafted in rounds 1-7 (picks 1-224)
        filtered_df = filtered_df[
            (filtered_df["PredictedDraftPosition"] > 0) & 
            (filtered_df["PredictedDraftPosition"] <= 224)
        ]

        # Apply sorting
        if sort_by:
            col, direction = sort_by.rsplit("_", 1)
            ascending = direction == "asc"
            if col in filtered_df.columns:
                filtered_df = filtered_df.sort_values(by=col, ascending=ascending)
        else:
            # Default sort by predicted draft position (highest picks first)
            filtered_df = filtered_df.sort_values(
                by="PredictedDraftPosition",
                ascending=True,
                na_position="last"
            )

        # Create results info text with more debug information
        results_info = f"Found {len(filtered_df)} prospects"
        if position and position != "All":
            results_info += f" in {position}"
        if school and school != "All":
            results_info += f" from {school}"
        if len(filtered_df) == 0:
            results_info += " (Note: Try adjusting your filters)"

        # Calculate total drafted
        total_drafted = len(filtered_df[filtered_df["PredictedDraftPosition"] > 0])

        return (
            filtered_df.to_dict("records"),
            results_info,
            str(len(filtered_df)),
            str(total_drafted),
        )

    @app.callback(
        [
            Output("players-container", "children"),
            Output("current-page", "data"),
            Output("total-pages", "data"),
            Output("pagination-display", "children"),
        ],
        [
            Input("filtered-data", "data"),
            Input("page-prev", "n_clicks"),
            Input("page-next", "n_clicks"),
            Input("page-first", "n_clicks"),
            Input("page-last", "n_clicks"),
            Input("page-input", "value"),
        ],
        [State("current-page", "data")]
    )
    def update_players_display(
        data, prev_clicks, next_clicks, first_clicks, last_clicks, 
        page_input, current_page
    ):
        if not data:
            return (
                html.Div("No players match your criteria. Try adjusting your filters."),
                1,
                1,
                html.Div("Page 1 of 1")
            )

        # Convert data back to dataframe
        filtered_df = pd.DataFrame(data)

        # Initialize current_page if None
        if current_page is None:
            current_page = 1

        # Determine which button was clicked
        ctx = dash.callback_context
        if ctx.triggered:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            # Handle navigation buttons
            if trigger_id == "page-prev" and current_page > 1:
                current_page -= 1
            elif trigger_id == "page-next":
                current_page += 1
            elif trigger_id == "page-first":
                current_page = 1
            elif trigger_id == "page-last":
                current_page = float("inf")  # Will be adjusted below
            elif trigger_id == "page-input" and page_input:
                try:
                    current_page = int(page_input)
                except:
                    pass

        # Determine pagination
        page_size = 10
        total_pages = max(1, (len(filtered_df) + page_size - 1) // page_size)

        # Make sure current_page is valid
        current_page = min(max(1, current_page), total_pages)

        # Get players for current page
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))
        page_data = filtered_df.iloc[start_idx:end_idx]

        # Create player cards
        player_cards = []
        for _, player in page_data.iterrows():
            # Create initials for avatar
            name_parts = str(player["name"]).split()
            initials = (
                "".join([name[0] for name in name_parts[:2]]).upper()
                if name_parts
                else "?"
            )

            # Determine if player is projected to be drafted
            draft_status = ""
            draft_badge_class = "badge-custom"
            if player.get("PredictedDraftPosition", 0) > 0:
                round_num = (player["PredictedDraftPosition"] // 32) + 1
                pick_num = player["PredictedDraftPosition"] % 32 or 32
                draft_status = f"Round {round_num}, Pick {pick_num}"

                # Color-code by round
                if round_num == 1:
                    draft_badge_class += " badge-high"
                elif round_num <= 3:
                    draft_badge_class += " badge-medium"
                else:
                    draft_badge_class += " badge-low"
            else:
                draft_status = "Undrafted"

            # Create player card
            player_card = html.Div(
                [
                    # Avatar and basic info
                    html.Div([html.Div(initials, className="player-avatar")]),
                    # Player details
                    html.Div(
                        [
                            # Name and basic info
                            html.H5(player["name"], className="player-name"),
                            html.P(
                                [
                                    f"{player['position_group']} • {player['readable_height']} • {int(player['weight_clean']) if not pd.isna(player.get('weight_clean')) else 'N/A'}lbs"
                                ],
                                className="player-position",
                            ),
                            html.P(player["school"], className="player-position"),
                            # Draft position (more prominent)
                            html.Div(
                                [
                                    html.Span(
                                        draft_status,
                                        className=draft_badge_class + " fw-bold",
                                    )
                                ],
                                className="mt-2 mb-2",
                            ),
                            # Badges
                            html.Div(
                                [
                                    html.Span(
                                        f"Weight: {int(player['weight_clean']) if not pd.isna(player.get('weight_clean')) else 'N/A'}lbs",
                                        className="badge-custom",
                                    ),
                                    html.Span(
                                        f"Arms: {player['arm_length']/100}\""
                                        if not pd.isna(player.get("arm_length"))
                                        else "",
                                        className="badge-custom",
                                    ),
                                    *[
                                        html.Span(
                                            f"40yd: {player['40_yard_dash']}s",
                                            className="badge-custom",
                                        )
                                        if not pd.isna(player.get("40_yard_dash"))
                                        else None,
                                        html.Span(
                                            f"Vert: {player['vertical']}\"",
                                            className="badge-custom",
                                        )
                                        if not pd.isna(player.get("vertical"))
                                        else None,
                                    ],
                                ],
                                className="mt-2 mb-3",
                            ),
                            # Action buttons
                            html.Div(
                                [
                                    dbc.Button(
                                        "Select",
                                        id={
                                            "type": "select-player-btn",
                                            "index": str(player["name"]),
                                        },
                                        color="primary",
                                        size="sm",
                                        className="me-2",
                                    ),
                                    dbc.Button(
                                        "View Details",
                                        id={
                                            "type": "view-player-btn",
                                            "index": str(player["name"]),
                                        },
                                        color="outline-primary",
                                        size="sm",
                                    ),
                                ],
                                className="mt-2",
                            ),
                        ],
                        className="ms-3 flex-grow-1",
                    ),
                ],
                className="player-card",
            )

            player_cards.append(player_card)

        # Create pagination display
        pagination_display = html.Div([
            f"Page {current_page} of {total_pages}"
        ], className="text-center fw-bold")
            
        return html.Div(player_cards), current_page, total_pages, pagination_display

    @app.callback(
        [
            Output("position-distribution", "figure"),
            Output("school-distribution", "figure"),
        ],
        Input("filtered-data", "data"),
    )
    def update_visualizations(data):
        if not data:
            # Return empty figures if no data
            empty_fig = {
                "data": [],
                "layout": {
                    "title": "No data available",
                    "height": 300,
                },
            }
            return empty_fig, empty_fig

        # Convert data back to dataframe
        filtered_df = pd.DataFrame(data)

        # Position distribution chart
        position_counts = filtered_df["position_group"].value_counts().reset_index()
        position_counts.columns = ["Position", "Count"]

        position_fig = px.bar(
            position_counts,
            x="Position",
            y="Count",
            color="Count",
            color_continuous_scale="Blues",
            title="Players by Position Group",
            template="plotly_white",
        )

        position_fig.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=50, b=40),
            coloraxis_showscale=False,
            xaxis_title="",
            yaxis_title="Number of Players",
        )

        # School distribution chart (top 10)
        school_counts = filtered_df["school"].value_counts().nlargest(10).reset_index()
        school_counts.columns = ["School", "Count"]

        school_fig = px.bar(
            school_counts,
            x="Count",
            y="School",
            orientation="h",
            color="Count",
            color_continuous_scale="Greens",
            title="Top 10 Schools by Player Count",
            template="plotly_white",
        )

        school_fig.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=50, b=40),
            coloraxis_showscale=False,
            xaxis_title="Number of Players",
            yaxis_title="",
        )

        return position_fig, school_fig

    @app.callback(
        [
            Output("comparison-section", "children"),
            Output("comparison-section", "className"),
        ],
        [Input("compare-players-btn", "n_clicks")],
        [State("selected-players", "data")],
    )
    def show_player_comparison(n_clicks, selected_players):
        if not n_clicks or not selected_players or len(selected_players) < 2:
            return html.Div(), "d-none"

        # Create comparison table
        comparison_data = []
        metrics = [
            ("Position", "position_group"),
            ("School", "school"),
            ("Height", "readable_height"),
            ("Weight", "weight_clean"),
            ("40 Yard", "40_yard_dash"),
            ("Vertical", "vertical"),
            ("Arm Length", "arm_length"),
            ("Draft Position", "PredictedDraftPosition"),
        ]

        # Create table header
        header_row = [html.Th("Metric")] + [
            html.Th(player["name"]) for player in selected_players
        ]
        comparison_data.append(html.Tr(header_row))

        # Create table rows
        for metric_name, metric_key in metrics:
            row_cells = [html.Td(metric_name)]
            
            # Get values for each player
            values = []
            for player in selected_players:
                value = player.get(metric_key)
                
                # Format the value based on the metric
                if metric_key == "weight_clean":
                    formatted_value = f"{int(value)}lbs" if pd.notna(value) else "N/A"
                elif metric_key == "40_yard_dash":
                    formatted_value = f"{value}s" if pd.notna(value) else "N/A"
                elif metric_key == "vertical":
                    formatted_value = f"{value}\"" if pd.notna(value) else "N/A"
                elif metric_key == "arm_length":
                    formatted_value = f"{value/100}\"" if pd.notna(value) else "N/A"
                elif metric_key == "PredictedDraftPosition":
                    if pd.notna(value) and value > 0:
                        round_num = (value // 32) + 1
                        pick_num = value % 32 or 32
                        formatted_value = f"Round {round_num}, Pick {pick_num}"
                    else:
                        formatted_value = "Undrafted"
                else:
                    formatted_value = str(value) if pd.notna(value) else "N/A"
                
                values.append(formatted_value)
            
            # Add cells to row with highlighting for best values
            for value in values:
                row_cells.append(html.Td(value))
            
            comparison_data.append(html.Tr(row_cells))

        comparison_content = html.Div(
            [
                html.H3("Player Comparison", className="mb-4"),
                dbc.Table(
                    comparison_data,
                    bordered=True,
                    hover=True,
                    striped=True,
                    className="comparison-table",
                ),
            ],
            className="mt-4",
        )

        return comparison_content, ""  # Empty className to show the section

    @app.callback(
        [
            Output("selected-players", "data"),
            Output("selected-players-list", "children"),
            Output("compare-players-btn", "disabled"),
        ],
        [
            Input(
                {"type": "select-player-btn", "index": dash.dependencies.ALL},
                "n_clicks",
            ),
            Input(
                {"type": "remove-selected", "index": dash.dependencies.ALL},
                "n_clicks",
            ),
        ],
        [State("filtered-data", "data"), State("selected-players", "data")],
    )
    def update_selected_players(n_clicks_list, remove_clicks_list, filtered_data, current_selected):
        if not current_selected:
            current_selected = []

        # Determine which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_selected, "No players selected", True

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        clicked_player = None

        try:
            button_id = json.loads(triggered_id)
            clicked_player = button_id["index"]
        except:
            return current_selected, "Error processing selection", True

        # Check if this is a remove action
        is_remove = triggered_id.startswith('{"type": "remove-selected"')
        
        if is_remove:
            # Remove player if already selected
            updated_selected = [p for p in current_selected if p["name"] != clicked_player]
        else:
            # Find the player in the filtered data
            filtered_df = pd.DataFrame(filtered_data)
            player_data = filtered_df[filtered_df["name"] == clicked_player].to_dict("records")

            if not player_data:
                return current_selected, "Selected player not found", True

            # Add player if not already selected (max 3)
            player_data = player_data[0]
            current_names = [p["name"] for p in current_selected]

            if clicked_player in current_names:
                # Remove player if already selected
                updated_selected = [p for p in current_selected if p["name"] != clicked_player]
            else:
                # Add player if not already selected (max 3)
                if len(current_selected) < 3:
                    updated_selected = current_selected + [player_data]
                else:
                    # Replace the first player if already at max
                    updated_selected = current_selected[1:] + [player_data]

        # Create the selected players list display
        selected_display = []
        for player in updated_selected:
            # Create the player badge with delete button
            player_badge = dbc.Badge(
                [
                    player["name"],
                    html.I(
                        className="fas fa-times ms-2",
                        id={"type": "remove-selected", "index": player["name"]},
                    ),
                ],
                color="primary",
                className="me-1 mb-2 p-2",
            )
            selected_display.append(player_badge)

        if not selected_display:
            selected_display = "No players selected"
            compare_disabled = True
        else:
            compare_disabled = len(updated_selected) < 2

        return updated_selected, selected_display, compare_disabled
