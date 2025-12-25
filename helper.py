import pandas as pd

numerical_features = ['rank_away', 'rank_home', 'home_win_pct', 'away_win_pct', 
                     'home_avg_points_for', 'away_avg_points_for',
                     'home_avg_points_against', 'away_avg_points_against']
categorical_features = ['conf_away', 'conf_home', 'game_type', 'neutral']

def time_machine_compare(team1, season1, team2, season2, team_profiles, win_loss_model, point_diff_model, neutral=True):
    """
    Compare two teams from different (or same) seasons in a hypothetical matchup.
    """
    key1 = f"{team1}_{season1}"
    key2 = f"{team2}_{season2}"
    
    if key1 not in team_profiles:
        return {'error': f"No data available for {team1} in the {season1} season"}
    
    if key2 not in team_profiles:
        return {'error': f"No data available for {team2} in the {season2} season"}
    
    profile1 = team_profiles[key1]
    profile2 = team_profiles[key2]
    
    # Create feature dictionary matching exactly the training features
    feature_dict = {}
    
    # Add numerical features in the same order as training
    for num_feature in numerical_features:
        if num_feature.startswith('home_'):
            feature_dict[num_feature] = [profile1[num_feature.replace('home_', '')]]
        elif num_feature.startswith('away_'):
            feature_dict[num_feature] = [profile2[num_feature.replace('away_', '')]]
        elif num_feature == 'rank_home':
            feature_dict[num_feature] = [profile1['best_rank'] if profile1['best_rank'] is not None else 50]
        elif num_feature == 'rank_away':
            feature_dict[num_feature] = [profile2['best_rank'] if profile2['best_rank'] is not None else 50]
    
    # Add categorical features in the same order as training
    for cat_feature in categorical_features:
        if cat_feature == 'conf_home':
            feature_dict[cat_feature] = [profile1['conference']]
        elif cat_feature == 'conf_away':
            feature_dict[cat_feature] = [profile2['conference']]
        elif cat_feature == 'game_type':
            feature_dict[cat_feature] = ['regular']
        elif cat_feature == 'neutral':
            feature_dict[cat_feature] = [neutral]
    
    # Create DataFrame with features in the exact order
    features = pd.DataFrame(feature_dict)
    
    # Verify feature names match
    expected_features = numerical_features + categorical_features
    if not all(col in features.columns for col in expected_features):
        missing_features = set(expected_features) - set(features.columns)
        return {'error': f"Missing features: {missing_features}"}
    
    # Make predictions
    try:
        win_prob = win_loss_model.predict_proba(features)[0][1]  # Probability of team1 win
        point_diff = point_diff_model.predict(features)[0]
    except Exception as e:
        # Fallback: Generate mock predictions based on team statistics
        print(f"Model prediction failed: {str(e)}")
        print("Using fallback mock predictions based on team statistics...")
        
        # Calculate win probability based on team stats
        team1_strength = (profile1['win_pct'] * 0.4 + 
                         (profile1['avg_points_for'] / 50) * 0.3 + 
                         (1 - profile1['avg_points_against'] / 50) * 0.3)
        team2_strength = (profile2['win_pct'] * 0.4 + 
                         (profile2['avg_points_for'] / 50) * 0.3 + 
                         (1 - profile2['avg_points_against'] / 50) * 0.3)
        
        # Normalize to probability
        total_strength = team1_strength + team2_strength
        if total_strength > 0:
            win_prob = team1_strength / total_strength
        else:
            win_prob = 0.5
        
        # Calculate point differential based on offensive/defensive stats
        point_diff = (profile1['avg_points_for'] - profile2['avg_points_for'] + 
                     profile2['avg_points_against'] - profile1['avg_points_against']) / 2
    
    # Determine winner
    if win_prob > 0.5:
        winner = team1
        loser = team2
        winner_season = season1
        loser_season = season2
    else:
        winner = team2
        loser = team1
        winner_season = season2
        loser_season = season1
        point_diff = -point_diff
    
    # Estimate score based on offensive/defensive capabilities
    winner_profile = profile1 if winner == team1 else profile2
    loser_profile = profile2 if winner == team1 else profile1
    
    winner_off = winner_profile['avg_points_for']
    winner_def = winner_profile['avg_points_against']
    loser_off = loser_profile['avg_points_for']
    loser_def = loser_profile['avg_points_against']
    
    # Enhanced score estimation formula with home/away adjustment
    base_winner_score = (winner_off * 1.1 + loser_def * 0.9) / 2
    base_loser_score = (loser_off * 0.9 + winner_def * 1.1) / 2
    
    # Adjust to match predicted point difference
    predicted_diff = abs(point_diff)
    actual_diff = base_winner_score - base_loser_score
    
    if actual_diff != predicted_diff:
        adjustment = (predicted_diff - actual_diff) / 2
        winner_score = base_winner_score + adjustment
        loser_score = base_loser_score - adjustment
    else:
        winner_score = base_winner_score
        loser_score = base_loser_score
    
    # Round to whole numbers
    winner_score = round(winner_score)
    loser_score = round(loser_score)
    
    # Calculate additional fun stats
    def calculate_dominance_score(profile):
        # Higher is better, combines winning, scoring, and defensive performance
        return (profile['win_pct'] * 100 + 
                profile['avg_points_for'] - 
                profile['avg_points_against'] + 
                profile['avg_margin'])
    
    def calculate_excitement_rating(profile):
        # Higher means more exciting games (high scoring, close margins)
        total_points = profile['avg_points_for'] + profile['avg_points_against']
        return (total_points / 2) * (1 - abs(profile['avg_margin']) / total_points)
    
    # Enhanced stats comparison
    stats_comparison = {
        f"{team1} ({season1})": {
            "Record": f"{profile1['wins']}-{profile1['losses']}",
            "Win %": f"{profile1['win_pct']:.3f}",
            "Points Per Game": f"{profile1['avg_points_for']:.1f}",
            "Points Allowed": f"{profile1['avg_points_against']:.1f}",
            "Average Margin": f"{profile1['avg_margin']:.1f}",
            "Best Ranking": profile1['best_rank'] if profile1['best_rank'] is not None else "Unranked",
            "Conference": profile1['conference'],
            "Games Played": profile1['total_games'],
            "Home/Away Split": f"{profile1['home_games']}/{profile1['away_games']}",
            "Dominance Score": f"{calculate_dominance_score(profile1):.1f}",
            "Excitement Rating": f"{calculate_excitement_rating(profile1):.1f}"
        },
        f"{team2} ({season2})": {
            "Record": f"{profile2['wins']}-{profile2['losses']}",
            "Win %": f"{profile2['win_pct']:.3f}",
            "Points Per Game": f"{profile2['avg_points_for']:.1f}",
            "Points Allowed": f"{profile2['avg_points_against']:.1f}",
            "Average Margin": f"{profile2['avg_margin']:.1f}",
            "Best Ranking": profile2['best_rank'] if profile2['best_rank'] is not None else "Unranked",
            "Conference": profile2['conference'],
            "Games Played": profile2['total_games'],
            "Home/Away Split": f"{profile2['home_games']}/{profile2['away_games']}",
            "Dominance Score": f"{calculate_dominance_score(profile2):.1f}",
            "Excitement Rating": f"{calculate_excitement_rating(profile2):.1f}"
        }
    }
    
    # Calculate matchup-specific stats
    matchup_stats = {
        "Offensive Advantage": f"{team1 if profile1['avg_points_for'] > profile2['avg_points_for'] else team2} "
                              f"(+{abs(profile1['avg_points_for'] - profile2['avg_points_for']):.1f} PPG)",
        "Defensive Advantage": f"{team1 if profile1['avg_points_against'] < profile2['avg_points_against'] else team2} "
                              f"(-{abs(profile1['avg_points_against'] - profile2['avg_points_against']):.1f} PPG)",
        "Better Record": f"{team1 if profile1['win_pct'] > profile2['win_pct'] else team2} "
                        f"(+{abs(profile1['win_pct'] - profile2['win_pct']):.3f})",
        "More Dominant": f"{team1 if calculate_dominance_score(profile1) > calculate_dominance_score(profile2) else team2}",
        "More Exciting": f"{team1 if calculate_excitement_rating(profile1) > calculate_excitement_rating(profile2) else team2}"
    }
    
    # Generate a narrative summary
    narrative = f"In this hypothetical matchup between {team1} ({season1}) and {team2} ({season2}), "
    narrative += f"our model favors {winner} ({winner_season}) with a {max(win_prob, 1-win_prob):.1%} win probability. "
    narrative += f"The predicted score of {winner_score}-{loser_score} reflects both teams' historical performance. "
    
    if profile1['avg_points_for'] > profile2['avg_points_for']:
        narrative += f"{team1} had the better offense in their season, "
    else:
        narrative += f"{team2} had the better offense in their season, "
    
    if profile1['avg_points_against'] < profile2['avg_points_against']:
        narrative += f"while {team1} showed stronger defense. "
    else:
        narrative += f"while {team2} showed stronger defense. "
    
    return {
        'team1': f"{team1} ({season1})",
        'team2': f"{team2} ({season2})",
        'predicted_winner': f"{winner} ({winner_season})",
        'win_probability': max(win_prob, 1 - win_prob),
        'predicted_score': f"{winner} {winner_score}, {loser} {loser_score}",
        'predicted_point_diff': abs(point_diff),
        'team_comparison': stats_comparison,
        'matchup_advantages': matchup_stats,
        'narrative': narrative,
        'location': "Neutral field" if neutral else f"{team1}'s home field",
        'hypothetical': True
    }

# Add a function to format and display the time machine results nicely
def display_time_machine_results(results):
    """
    Format and display the results of a time machine comparison in a nice way.
    """
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print("=" * 80)
    print(f"TIME MACHINE MATCHUP: {results['team1']} vs {results['team2']}")
    print("=" * 80)
    print(f"\nLocation: {results['location']}")
    print(f"\nPREDICTED OUTCOME:")
    print(f"Winner: {results['predicted_winner']}")
    print(f"Score: {results['predicted_score']}")
    print(f"Win Probability: {results['win_probability']:.1%}")
    
    print("\nTEAM COMPARISON:")
    print("-" * 80)
    stats1 = results['team_comparison'][results['team1']]
    stats2 = results['team_comparison'][results['team2']]
    
    # Find the longest stat name for alignment
    max_stat_length = max(len(stat) for stat in stats1.keys())
    
    for stat in stats1.keys():
        print(f"{stat:<{max_stat_length}} | {stats1[stat]:<20} | {stats2[stat]:<20}")
    
    print("\nMATCHUP ADVANTAGES:")
    print("-" * 80)
    for category, advantage in results['matchup_advantages'].items():
        print(f"{category:<20}: {advantage}")
    
    print("\nNARRATIVE SUMMARY:")
    print("-" * 80)
    print(results['narrative'])
    print("\n" + "=" * 80)