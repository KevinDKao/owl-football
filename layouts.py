from dash import html, dcc
import dash_bootstrap_components as dbc
import dash
import dash.dash_table

# List of available teams from example.py
AVAILABLE_TEAMS = [
    "Abilene Christian", "Air Force", "Akron", "Alabama", "Appalachian State", "Arizona", "Arizona State",
    "Arkansas", "Arkansas State", "Army", "Auburn", "Austin Peay", "BYU", "Ball State", "Baylor",
    "Boise State", "Boston College", "Bowling Green", "Buffalo", "California", "Campbell", "Central Michigan",
    "Charlotte", "Cincinnati", "Citadel", "Clemson", "Coastal Carolina", "Colorado", "Colorado State",
    "Connecticut", "Duke", "East Carolina", "Eastern Kentucky", "Eastern Michigan", "FIU", "Florida",
    "Florida A&M", "Florida Atlantic", "Florida State", "Fresno State", "Gardner Webb", "Georgia",
    "Georgia Southern", "Georgia State", "Georgia Tech", "Hawaii", "Houston", "Houston Baptist", "Idaho",
    "Illinois", "Indiana", "Iowa", "Iowa State", "Jacksonville State", "James Madison", "Kansas",
    "Kansas State", "Kennesaw State", "Kent State", "Kentucky", "LSU", "Liberty", "Louisiana Tech",
    "Louisville", "Marshall", "Maryland", "Massachusetts", "Memphis", "Miami (FL)", "Miami (OH)",
    "Michigan", "Michigan State", "Middle Tennessee", "Minnesota", "Mississippi State", "Missouri",
    "NC State", "Navy", "Nebraska", "Nevada", "New Mexico", "New Mexico State", "Nicholls State",
    "Norfolk State", "North Carolina", "North Texas", "Northern Illinois", "Northwestern",
    "Northwestern State", "Notre Dame", "Ohio", "Ohio State", "Oklahoma", "Oklahoma State",
    "Old Dominion", "Ole Miss", "Oregon", "Oregon State", "Penn State", "Pittsburgh", "Portland State",
    "Presbyterian", "Purdue", "Rice", "Rutgers", "SMU", "Sam Houston State", "San Diego State",
    "San Jose State", "Savannah State", "South Alabama", "South Carolina", "Southeastern Louisiana",
    "Southern Miss", "Stanford", "Stephen F. Austin", "Syracuse", "TCU", "Temple", "Tennessee",
    "Tennessee-Martin", "Texas", "Texas A&M", "Texas State", "Texas Tech", "Toledo", "Troy", "Tulane",
    "Tulsa", "UAB", "UCF", "UCLA", "UL-Lafayette", "UL-Monroe", "UNLV", "USC", "USF", "UTEP", "UTSA",
    "Utah", "Utah State", "Vanderbilt", "Virginia", "Virginia Tech", "Wake Forest", "Washington",
    "Washington State", "West Virginia", "Western Carolina", "Western Kentucky", "Western Michigan",
    "Wisconsin", "Wyoming"
]

def create_time_machine_tab():
    """Create the Time Machine tab content"""
    return html.Div(
        [
            html.H2("Time Machine", className="text-center mb-4"),
            html.P(
                "Compare teams from different seasons in a hypothetical matchup!",
                className="text-center mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Team 1", className="mb-3"),
                            dbc.Select(
                                id="team1-input",
                                options=[{"label": team, "value": team} for team in AVAILABLE_TEAMS],
                                value="Michigan",
                                className="mb-3",
                            ),
                            dbc.Select(
                                id="season1-select",
                                options=[
                                    {"label": str(year), "value": year}
                                    for year in range(2003, 2025)
                                ],
                                value=2023,
                                className="mb-3",
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H5("Team 2", className="mb-3"),
                            dbc.Select(
                                id="team2-input",
                                options=[{"label": team, "value": team} for team in AVAILABLE_TEAMS],
                                value="Georgia",
                                className="mb-3",
                            ),
                            dbc.Select(
                                id="season2-select",
                                options=[
                                    {"label": str(year), "value": year}
                                    for year in range(2003, 2025)
                                ],
                                value=2022,
                                className="mb-3",
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Checkbox(
                                id="neutral-site",
                                label="Neutral Site Game",
                                value=True,
                                className="mb-3",
                            ),
                            dbc.Button(
                                "Run Time Machine",
                                id="run-time-machine",
                                color="primary",
                                className="w-100",
                            ),
                        ],
                        width=12,
                    ),
                ],
            ),
            html.Div(id="time-machine-results", className="mt-4"),
        ],
        className="time-machine-section p-4 border rounded",
    )

def create_draft_predictions_tab():
    """Create the NFL Draft Predictions tab content"""
    return html.Div(
        [
            html.Div(
                [
                    html.H2("NFL Draft Predictions", className="text-center mb-4"),
                    html.P(
                        "Explore prospects and their predicted draft positions",
                        className="text-center mb-4 text-muted",
                    ),
                ],
                className="mb-4",
            ),
            # Filters section
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H5("Filter by Position", className="mb-2"),
                                    dcc.Dropdown(
                                        id="position-filter",
                                        options=[
                                            {"label": "All Positions", "value": "all"},
                                            {"label": "Quarterback", "value": "QB"},
                                            {"label": "Running Back", "value": "RB"},
                                            {"label": "Wide Receiver", "value": "WR"},
                                            {"label": "Tight End", "value": "TE"},
                                            {"label": "Offensive Line", "value": "OL"},
                                            {"label": "Defensive Line", "value": "DL"},
                                            {"label": "Linebacker", "value": "LB"},
                                            {"label": "Defensive Back", "value": "DB"},
                                        ],
                                        value="all",
                                        className="mb-3",
                                        clearable=False,
                                        searchable=True,
                                    ),
                                ],
                                className="p-3 bg-light rounded",
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H5("Filter by School", className="mb-2"),
                                    dcc.Dropdown(
                                        id="school-filter",
                                        options=[{"label": "All Schools", "value": "all"}],
                                        value="all",
                                        className="mb-3",
                                        clearable=False,
                                        searchable=True,
                                    ),
                                ],
                                className="p-3 bg-light rounded",
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H5("Draft Status", className="mb-2"),
                                    dcc.Dropdown(
                                        id="draft-status-filter",
                                        options=[
                                            {"label": "Drafted Only", "value": "Drafted"},
                                            {"label": "All Players", "value": "all"},
                                        ],
                                        value="Drafted",
                                        className="mb-3",
                                        clearable=False,
                                    ),
                                ],
                                className="p-3 bg-light rounded",
                            ),
                        ],
                        width=4,
                    ),
                ],
                className="mb-4",
            ),
            # Player cards container
            html.Div(
                id="player-cards-container",
                className="row row-cols-1 row-cols-md-2 row-cols-lg-3 g-4",
            ),
        ],
        className="draft-predictions-section p-4",
    )

# Create the app layout
def create_layout():
    """
    Create the main layout of the application
    """
    layout = html.Div(
        [
            # Header
            html.Div(
                [
                    dbc.Container(
                        [
                            html.H1(
                                "Owl Football! D2K 2025",
                                className="header-title text-center",
                            ),
                            html.H4(
                                "Compare teams across time and explore draft predictions",
                                className="header-subtitle text-center mt-2",
                            ),
                        ],
                        fluid=True,
                    )
                ],
                className="header-section",
            ),
            # Main content
            dbc.Container(
                [
                    # Tabs
                    dbc.Tabs(
                        [
                            dbc.Tab(
                                create_time_machine_tab(),
                                label="Time Machine",
                                tab_id="time-machine",
                                active_tab_class_name="fw-bold",
                            ),
                            dbc.Tab(
                                create_draft_predictions_tab(),
                                label="NFL Draft Predictions",
                                tab_id="draft-predictions",
                                active_tab_class_name="fw-bold",
                            ),
                        ],
                        id="tabs",
                        active_tab="time-machine",
                        className="mb-4",
                    ),
                ],
                fluid=True,
            ),
        ],
        className="dashboard-container",
    )

    return layout
