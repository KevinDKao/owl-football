import dash
from dash import html
import dash_bootstrap_components as dbc

from layouts import create_layout
from callbacks import register_callbacks
from data import load_data, get_positions_and_schools

# Initialize the Dash app with a modern light theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server

# Custom HTML template with the CSS styling
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Owl Football! D2K 2025</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


def init_app():
    """
    Initialize the app with data and callbacks
    """
    # Load and process data for draft predictions
    app.df = load_data()
    positions, schools = get_positions_and_schools(app.df)

    # Set app layout
    app.layout = create_layout()

    # Register callbacks
    register_callbacks(app)

    return app


# Initialize app
init_app()

# Entry point
if __name__ == "__main__":
    app.run(debug=False, port=8080)
