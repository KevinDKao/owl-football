import pandas as pd
import plotly.express as px
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import json
import pickle
import os
import traceback
from helper import time_machine_compare
from data import load_data
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_callbacks(app):
    """
    Register all callback functions for the app
    """

    # Load time machine models and data
    try:
        # Load models from root directory
        with open('app/win_loss_model.pkl', 'rb') as f:
            win_loss_model = pickle.load(f)
        with open('app/point_diff_model.pkl', 'rb') as f:
            point_diff_model = pickle.load(f)
        with open('app/team_season_profiles.pkl', 'rb') as f:
            team_profiles = pickle.load(f)
            
        print("Successfully loaded all model files")
        
    except Exception as e:
        print(f"Error loading time machine models: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        win_loss_model = None
        point_diff_model = None
        team_profiles = None

    @app.callback(
        Output("time-machine-results", "children"),
        [Input("run-time-machine", "n_clicks")],
        [
            State("team1-input", "value"),
            State("season1-select", "value"),
            State("team2-input", "value"),
            State("season2-select", "value"),
            State("neutral-site", "value"),
        ],
    )
    def run_time_machine(n_clicks, team1, season1, team2, season2, neutral):
        if not n_clicks or not team1 or not team2:
            return html.Div("Please enter both teams to compare.")

        if not all([win_loss_model, point_diff_model, team_profiles]):
            return html.Div(
                "Error: Time machine models not loaded. Please check the model files.",
                className="alert alert-danger"
            )

        try:
            results = time_machine_compare(
                team1, season1, team2, season2,
                team_profiles, win_loss_model, point_diff_model,
                neutral=neutral
            )

            if 'error' in results:
                return html.Div(
                    f"Error: {results['error']}",
                    className="alert alert-danger"
                )

            # Create the results display
            return html.Div(
                [
                    html.H3(f"Time Machine Matchup: {results['team1']} vs {results['team2']}", className="text-center mb-4"),
                    html.P(f"Location: {results['location']}", className="text-center mb-3"),
                    
                    # Predicted Outcome
                    html.Div(
                        [
                            html.H4("Predicted Outcome", className="mb-3"),
                            html.P(f"Winner: {results['predicted_winner']}"),
                            html.P(f"Score: {results['predicted_score']}"),
                            html.P(f"Win Probability: {results['win_probability']:.1%}"),
                        ],
                        className="mb-4 p-3 border rounded"
                    ),
                    
                    # Team Comparison
                    html.Div(
                        [
                            html.H4("Team Comparison", className="mb-3"),
                            dbc.Table(
                                [
                                    html.Thead(
                                        html.Tr(
                                            [
                                                html.Th("Stat"),
                                                html.Th(results['team1']),
                                                html.Th(results['team2'])
                                            ]
                                        )
                                    ),
                                    html.Tbody(
                                        [
                                            html.Tr(
                                                [
                                                    html.Td(stat),
                                                    html.Td(results['team_comparison'][results['team1']][stat]),
                                                    html.Td(results['team_comparison'][results['team2']][stat])
                                                ]
                                            )
                                            for stat in results['team_comparison'][results['team1']].keys()
                                        ]
                                    )
                                ],
                                bordered=True,
                                hover=True,
                                className="mb-4"
                            )
                        ],
                        className="mb-4 p-3 border rounded"
                    ),
                    
                    # Matchup Advantages
                    html.Div(
                        [
                            html.H4("Matchup Advantages", className="mb-3"),
                            dbc.ListGroup(
                                [
                                    dbc.ListGroupItem(
                                        f"{category}: {advantage}"
                                    )
                                    for category, advantage in results['matchup_advantages'].items()
                                ]
                            )
                        ],
                        className="mb-4 p-3 border rounded"
                    ),
                    
                    # Narrative Summary
                    html.Div(
                        [
                            html.H4("Narrative Summary", className="mb-3"),
                            html.P(results['narrative'])
                        ],
                        className="p-3 border rounded"
                    )
                ],
                className="time-machine-results"
            )

        except Exception as e:
            return html.Div(
                f"An error occurred: {str(e)}",
                className="alert alert-danger"
            )

    @callback(
        Output("player-cards-container", "children"),
        Output("school-filter", "options"),
        Input("position-filter", "value"),
        Input("school-filter", "value"),
        Input("draft-status-filter", "value"),
    )
    def update_player_cards(position_filter, school_filter, draft_status_filter):
        """Update the player cards based on filters"""
        # Load the draft predictions data
        df = pd.read_csv("cache.csv")
        
        # Convert height from inches to feet and inches
        def convert_height(height):
            if pd.isna(height):
                return "N/A"
            try:
                # Convert to string and handle decimal point
                height_str = str(int(height))
                feet = height_str[:-2]
                inches = height_str[-2:]
                return f"{feet}'{inches}\""
            except:
                return "N/A"
        
        df['readable_height'] = df['HEIGHT'].apply(convert_height)
        
        # Apply position filter
        if position_filter != "all":
            df = df[df["position_group"] == position_filter]
        
        # Apply school filter
        if school_filter != "all":
            df = df[df["SCHOOL:"] == school_filter]
        
        # Apply draft status filter
        if draft_status_filter != "all":
            df = df[df["DraftStatus"] == draft_status_filter]
        
        # Sort by draft position (highest to lowest)
        df = df.sort_values(by="predicted_draft_position", ascending=True)
        
        # Create cards for each player
        cards = []
        for _, player in df.iterrows():
            # Calculate draft round based on position
            draft_round = "Undrafted" if player["predicted_draft_position"] == 0 else f"Round {((player['predicted_draft_position'] - 1) // 32) + 1}"
            
            # Format 40 yard dash time
            forty_time = player['40 Yard Dash'] if pd.notna(player['40 Yard Dash']) else 'N/A'
            
            card = dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.H4(f"{player['NAME:']}", className="card-title mb-0"),
                            html.Small(f"{player['SCHOOL:']}", className="text-muted"),
                        ],
                        className="bg-primary text-white",
                    ),
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Strong("Position: "),
                                            html.Span(f"{player['position_group']}"),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Div(
                                        [
                                            html.Strong("Height/Weight: "),
                                            html.Span(f"{player['readable_height']} / {player['WEIGHT']} lbs"),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Div(
                                        [
                                            html.Strong("Draft Position: "),
                                            html.Span(f"{player['predicted_draft_position'] if player['predicted_draft_position'] > 0 else 'Undrafted'}"),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Div(
                                        [
                                            html.Strong("Draft Round: "),
                                            html.Span(draft_round),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Div(
                                        [
                                            html.Strong("40 Yard Dash: "),
                                            html.Span(forty_time),
                                        ],
                                        className="mb-2",
                                    ),
                                ],
                                className="player-stats",
                            ),
                        ],
                        className="p-3",
                    ),
                ],
                className="h-100 shadow-sm",
            )
            cards.append(dbc.Col(card, className="mb-4"))
        
        # Update school filter options
        schools = sorted(df["SCHOOL:"].unique())
        school_options = [{"label": "All Schools", "value": "all"}] + [
            {"label": school, "value": school} for school in schools
        ]
        
        return cards, school_options
