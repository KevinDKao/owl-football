import pandas as pd
import numpy as np

# Helper function to convert height from NFL format to readable format
def convert_height(height_value):
    if pd.isna(height_value):
        return "N/A"

    try:
        height_str = str(int(height_value))
        if len(height_str) < 4:
            return "N/A"

        feet = height_str[0]
        inches = height_str[1:3]
        return f"{feet}'{inches}\""
    except:
        return "N/A"


# Mock data for schools and players with more realistic names
schools = [
    "Alabama",
    "Georgia",
    "Ohio State",
    "Michigan",
    "LSU",
    "Clemson",
    "USC",
    "Texas",
    "Florida",
    "Notre Dame",
]

# List of realistic player names
first_names = [
    "Jackson",
    "Malik",
    "Trevon",
    "DeShawn",
    "Jayden",
    "Xavier",
    "Zach",
    "Tyler",
    "Caleb",
    "Isaiah",
    "Aidan",
    "Connor",
    "Bryce",
    "Justin",
    "Trevor",
    "Marcus",
    "Jamal",
    "Elijah",
    "Darius",
    "Cameron",
]
last_names = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Miller",
    "Davis",
    "Wilson",
    "Anderson",
    "Thomas",
    "Jackson",
    "White",
    "Harris",
    "Martin",
    "Thompson",
    "Moore",
    "Allen",
    "Young",
    "Wright",
    "Scott",
]

# Generate more realistic player data
def generate_mock_player_data():
    np.random.seed(42)  # For reproducibility
    players_data = {}

    for school in schools:
        players = []
        for i in range(5):
            name = f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
            position = np.random.choice(
                ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]
            )
            height_inches = np.random.randint(70, 78)
            height_ft = height_inches // 12
            height_in = height_inches % 12
            height = f"{height_ft}'{height_in}\""

            weight = np.random.randint(180, 340)
            speed = round(np.random.uniform(4.3, 5.0), 2)
            strength = np.random.randint(70, 99)
            agility = np.random.randint(70, 99)
            technique = np.random.randint(70, 99)
            football_iq = np.random.randint(70, 99)
            draft_probability = round(np.random.uniform(0.3, 0.95), 2)

            # Calculate a draft round prediction based on stats
            stats_avg = (speed * 20 + strength + agility + technique + football_iq) / 5
            if stats_avg > 90:
                draft_round = 1
            elif stats_avg > 85:
                draft_round = 2
            elif stats_avg > 80:
                draft_round = 3
            elif stats_avg > 75:
                draft_round = 4
            elif stats_avg > 70:
                draft_round = 5
            else:
                draft_round = np.random.randint(6, 8)

            # Generate team fits based on position
            if position == "QB":
                team_fits = np.random.choice(
                    ["Colts", "Broncos", "Raiders", "Falcons", "Saints"],
                    3,
                    replace=False,
                )
            elif position in ["RB", "WR", "TE"]:
                team_fits = np.random.choice(
                    ["Chiefs", "Bills", "Bengals", "Eagles", "Dolphins"],
                    3,
                    replace=False,
                )
            else:
                team_fits = np.random.choice(
                    ["Ravens", "Steelers", "49ers", "Cowboys", "Patriots"],
                    3,
                    replace=False,
                )

            players.append(
                {
                    "name": name,
                    "position": position,
                    "height": height,
                    "weight": f"{weight} lbs",
                    "speed": speed,
                    "strength": strength,
                    "agility": agility,
                    "technique": technique,
                    "football_iq": football_iq,
                    "draft_probability": draft_probability,
                    "draft_round": draft_round,
                    "team_fits": team_fits,
                    "initials": "".join([n[0] for n in name.split()]),
                }
            )

        players_data[school] = players

    return players_data
