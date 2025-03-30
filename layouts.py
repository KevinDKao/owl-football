from dash import html, dcc
import dash_bootstrap_components as dbc

# Create the app layout
def create_layout(positions, schools):
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
                                "2025 NFL Draft Predictions",
                                className="header-title text-center",
                            ),
                            html.H4(
                                "Explore prospects and their predicted draft positions",
                                className="header-subtitle text-center mt-2",
                            ),
                            # Stats row
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "0",
                                                        id="total-prospects",
                                                        className="stat-value",
                                                    ),
                                                    html.Div(
                                                        "Prospects",
                                                        className="stat-label",
                                                    ),
                                                ],
                                                className="stat-card",
                                            )
                                        ],
                                        width=12,
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        str(len(positions)),
                                                        className="stat-value",
                                                    ),
                                                    html.Div(
                                                        "Positions",
                                                        className="stat-label",
                                                    ),
                                                ],
                                                className="stat-card",
                                            )
                                        ],
                                        width=12,
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        str(len(schools)),
                                                        className="stat-value",
                                                    ),
                                                    html.Div(
                                                        "Schools",
                                                        className="stat-label",
                                                    ),
                                                ],
                                                className="stat-card",
                                            )
                                        ],
                                        width=12,
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "0",
                                                        id="total-drafted",
                                                        className="stat-value",
                                                    ),
                                                    html.Div(
                                                        "Projected Drafted",
                                                        className="stat-label",
                                                    ),
                                                ],
                                                className="stat-card",
                                            )
                                        ],
                                        width=12,
                                        md=3,
                                    ),
                                ],
                                className="mb-4 mt-4",
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
                    dbc.Row(
                        [
                            # Left sidebar with filters
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.H4("Filters", className="mb-3"),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Position Group",
                                                        className="fw-bold mb-2",
                                                    ),
                                                    dcc.RadioItems(
                                                        id="position-filter",
                                                        options=[
                                                            {"label": pos, "value": pos}
                                                            for pos in positions
                                                        ]
                                                        + [
                                                            {
                                                                "label": "All",
                                                                "value": "All",
                                                            }
                                                        ],
                                                        value="All",
                                                        className="position-filter-group mb-3",
                                                        inputStyle={
                                                            "margin-right": "5px"
                                                        },
                                                        labelStyle={
                                                            "margin-right": "12px",
                                                            "margin-bottom": "10px",
                                                            "display": "inline-block",
                                                        },
                                                        inline=True,
                                                        style={
                                                            "display": "flex",
                                                            "flex-wrap": "wrap",
                                                            "gap": "10px",
                                                        },
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "School",
                                                        className="fw-bold mb-2",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="school-filter",
                                                        options=[
                                                            {
                                                                "label": school,
                                                                "value": school,
                                                            }
                                                            for school in schools
                                                        ]
                                                        + [
                                                            {
                                                                "label": "All",
                                                                "value": "All",
                                                            }
                                                        ],
                                                        value="All",
                                                        clearable=False,
                                                        className="mb-3",
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Sort By",
                                                        className="fw-bold mb-2",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="sort-by",
                                                        options=[
                                                            {
                                                                "label": "Draft Position (Highest)",
                                                                "value": "PredictedDraftPosition_asc",
                                                            },
                                                            {
                                                                "label": "Draft Position (Lowest)",
                                                                "value": "PredictedDraftPosition_desc",
                                                            },
                                                            {
                                                                "label": "Name (A-Z)",
                                                                "value": "name_asc",
                                                            },
                                                            {
                                                                "label": "Name (Z-A)",
                                                                "value": "name_desc",
                                                            },
                                                            {
                                                                "label": "School (A-Z)",
                                                                "value": "school_asc",
                                                            },
                                                            {
                                                                "label": "School (Z-A)",
                                                                "value": "school_desc",
                                                            },
                                                            {
                                                                "label": "40 Yard (Fastest)",
                                                                "value": "40_yard_dash_asc",
                                                            },
                                                            {
                                                                "label": "Vertical (Highest)",
                                                                "value": "vertical_desc",
                                                            },
                                                            {
                                                                "label": "Weight (Heaviest)",
                                                                "value": "weight_clean_desc",
                                                            },
                                                            {
                                                                "label": "Weight (Lightest)",
                                                                "value": "weight_clean_asc",
                                                            },
                                                        ],
                                                        value="PredictedDraftPosition_asc",
                                                        clearable=False,
                                                        className="mb-3",
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Physical Attributes",
                                                        className="fw-bold mb-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Min Height",
                                                                        className="metric-label",
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="min-height",
                                                                        options=[
                                                                            {
                                                                                "label": "Any",
                                                                                "value": "Any",
                                                                            }
                                                                        ]
                                                                        + [
                                                                            {
                                                                                "label": f"{height}",
                                                                                "value": height,
                                                                            }
                                                                            for height in sorted(
                                                                                [
                                                                                    "5'10\"",
                                                                                    "5'11\"",
                                                                                    "6'0\"",
                                                                                    "6'1\"",
                                                                                    "6'2\"",
                                                                                    "6'3\"",
                                                                                    "6'4\"",
                                                                                    "6'5\"",
                                                                                    "6'6\"",
                                                                                    "6'7\"",
                                                                                ]
                                                                            )
                                                                        ],
                                                                        value="Any",
                                                                        clearable=False,
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Min Weight",
                                                                        className="metric-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="min-weight",
                                                                        type="number",
                                                                        placeholder="Min weight",
                                                                        min=150,
                                                                        max=350,
                                                                        step=1,
                                                                        value=None,
                                                                        className="form-control",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "40 Yard Dash (seconds)",
                                                                        className="metric-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="min-forty",
                                                                        type="number",
                                                                        placeholder="4.2 to 5.5 sec",
                                                                        min=4.2,
                                                                        max=5.5,
                                                                        step=0.01,
                                                                        value=None,
                                                                        className="form-control",
                                                                    ),
                                                                    html.Small(
                                                                        "Lower is better (4.2-5.5 sec)",
                                                                        className="text-muted d-block mt-1",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    html.Label(
                                                                        "Vertical Jump (inches)",
                                                                        className="metric-label",
                                                                    ),
                                                                    dcc.Input(
                                                                        id="min-vertical",
                                                                        type="number",
                                                                        placeholder="20 to 45 inches",
                                                                        min=20,
                                                                        max=45,
                                                                        step=0.5,
                                                                        value=None,
                                                                        className="form-control",
                                                                    ),
                                                                    html.Small(
                                                                        "Higher is better (20-45 inches)",
                                                                        className="text-muted d-block mt-1",
                                                                    ),
                                                                ],
                                                                width=6,
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "Apply Filters",
                                                        id="apply-filters",
                                                        color="primary",
                                                        className="w-100 mt-4",
                                                    ),
                                                ]
                                            ),
                                            html.Hr(),
                                            html.Div(
                                                [
                                                    html.H5(
                                                        "Selected Players",
                                                        className="mb-3",
                                                    ),
                                                    html.Div(
                                                        id="selected-players-list",
                                                        className="mb-3",
                                                    ),
                                                    dbc.Button(
                                                        "Compare Players",
                                                        id="compare-players-btn",
                                                        color="success",
                                                        className="w-100",
                                                        disabled=True,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="filter-card sticky-top-custom",
                                    )
                                ],
                                width=12,
                                md=4,
                                lg=3,
                            ),
                            # Main content area - Player listings and details
                            dbc.Col(
                                [
                                    # Results section
                                    html.Div(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.H3(
                                                                "Prospects",
                                                                className="mb-3",
                                                            ),
                                                            html.Div(
                                                                id="results-info",
                                                                className="text-muted mb-3",
                                                            ),
                                                        ],
                                                        width=7,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Input(
                                                                id="search-input",
                                                                placeholder="Search by name...",
                                                                type="text",
                                                                className="mb-3",
                                                            ),
                                                        ],
                                                        width=5,
                                                    ),
                                                ]
                                            ),
                                            # Player listing
                                            html.Div(
                                                id="players-container", className="mt-3"
                                            ),
                                            # Pagination
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    dbc.Button(
                                                                        "«",
                                                                        id="page-first",
                                                                        color="link",
                                                                        size="sm",
                                                                        className="me-1",
                                                                    ),
                                                                    dbc.Button(
                                                                        "<",
                                                                        id="page-prev",
                                                                        color="link",
                                                                        size="sm",
                                                                        className="me-1",
                                                                    ),
                                                                    html.Div(
                                                                        id="pagination-display",
                                                                        className="mx-2 fw-bold",
                                                                    ),
                                                                    html.Div(
                                                                        [
                                                                            dbc.Input(
                                                                                id="page-input",
                                                                                type="number",
                                                                                min=1,
                                                                                value=1,
                                                                                style={
                                                                                    "width": "70px"
                                                                                },
                                                                                className="mx-2",
                                                                            ),
                                                                        ],
                                                                        className="d-flex align-items-center",
                                                                    ),
                                                                    dbc.Button(
                                                                        ">",
                                                                        id="page-next",
                                                                        color="link",
                                                                        size="sm",
                                                                        className="ms-1",
                                                                    ),
                                                                    dbc.Button(
                                                                        "»",
                                                                        id="page-last",
                                                                        color="link",
                                                                        size="sm",
                                                                        className="ms-1",
                                                                    ),
                                                                ],
                                                                className="d-flex justify-content-center align-items-center mt-3",
                                                            ),
                                                            # Hidden elements to store pagination state
                                                            dcc.Store(
                                                                id="current-page",
                                                                data=1,
                                                            ),
                                                            dcc.Store(
                                                                id="total-pages", data=1
                                                            ),
                                                        ],
                                                        className="d-flex justify-content-center",
                                                    )
                                                ]
                                            ),
                                        ],
                                        className="mb-4",
                                    ),
                                    # Visualizations section
                                    html.Div(
                                        [
                                            html.H3(
                                                "Position Breakdown", className="mb-3"
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dcc.Graph(
                                                                id="position-distribution",
                                                                config={
                                                                    "displayModeBar": False
                                                                },
                                                            )
                                                        ],
                                                        width=12,
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dcc.Graph(
                                                                id="school-distribution",
                                                                config={
                                                                    "displayModeBar": False
                                                                },
                                                            )
                                                        ],
                                                        width=12,
                                                        md=6,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="mb-4",
                                    ),
                                    # Player comparisons and details section
                                    html.Div(
                                        id="player-details-section", className="d-none"
                                    ),
                                    # Comparison section
                                    html.Div(
                                        id="comparison-section", className="d-none"
                                    ),
                                ],
                                width=12,
                                md=8,
                                lg=9,
                            ),
                        ]
                    ),
                ],
                className="dashboard-container",
                fluid=True,
            ),
            # Footer
            html.Div(
                [
                    dbc.Container(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H5(
                                                "NFL Draft Predictor", className="mb-3"
                                            ),
                                            html.P(
                                                "A comprehensive dashboard for exploring 2025 NFL Draft prospects, their measurables, and predicted draft positions."
                                            ),
                                        ],
                                        width=12,
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H5("About", className="mb-3"),
                                            html.P(
                                                "This dashboard uses predictive models to generate draft projections based on player measurables and performance metrics."
                                            ),
                                        ],
                                        width=12,
                                        md=6,
                                    ),
                                ]
                            ),
                            html.Hr(className="bg-light"),
                            html.P(
                                "© 2025 NFL Draft Predictor • All Rights Reserved",
                                className="text-center mb-0",
                            ),
                        ],
                        fluid=True,
                    )
                ],
                className="footer",
            ),
            # Fixed comparison button
            html.Div(
                [html.I(className="fas fa-chart-bar compare-icon")],
                id="floating-compare-btn",
                className="compare-btn d-none",
            ),
            # Store components
            dcc.Store(id="filtered-data"),
            dcc.Store(id="selected-players", data=[]),
            dcc.Store(id="comparison-active", data=False),
        ]
    )

    return layout
