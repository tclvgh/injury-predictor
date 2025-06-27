from pybaseball import statcast

# MLB Pitching Logs from 2020 to 2024, Regular and Postseason

# 2020
# myStats = statcast(start_dt="2020-07-23", end_dt="2020-10-27")

# 2021
# myStats = statcast(start_dt="2021-04-01", end_dt="2021-11-02")

# 2022
# myStats = statcast(start_dt="2022-04-07", end_dt="2022-11-05")

# 2023
# myStats = statcast(start_dt="2023-03-30", end_dt="2023-11-01")

# 2024
myStats = statcast(start_dt="2024-03-20", end_dt="2024-10-30")

# print(myStats.columns)

myStatsData = myStats.drop(columns=['age_pit_legacy', 'at_bat_number', 'away_score', 'away_team', 'babip_value', 'balls', 'bat_score',
                      'bat_score_diff', 'bat_speed', 'bat_win_exp', 'batter', 'bb_type', 'break_angle_deprecated',
                      'break_length_deprecated', 'delta_home_win_exp', 'delta_pitcher_run_exp',
                      'delta_run_exp', 'estimated_ba_using_speedangle', 'estimated_slg_using_speedangle',
                      'estimated_woba_using_speedangle', 'fielder_2', 'fielder_3', 'fielder_4', 'fielder_5',
                      'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9', 'fld_score', 'hit_distance_sc',
                      'hit_location', 'home_score', 'home_score_diff', 'home_team', 'home_win_exp', 'hyper_speed',
                      'if_fielding_alignment', 'inning_topbot', 'launch_angle', 'launch_speed', 'launch_speed_angle',
                      'of_fielding_alignment', 'on_1b', 'on_2b', 'on_3b', 'outs_when_up', 'post_away_score',
                      'post_bat_score', 'post_fld_score', 'post_home_score', 'strikes', 'swing_length', 'umpire',
                      'woba_denom', 'woba_value', 'age_bat', 'age_bat_legacy', 'attack_angle', 'attack_direction',
                      'batter_days_since_prev_game', 'batter_days_until_next_game',
                      'intercept_ball_minus_batter_pos_x_inches', 'intercept_ball_minus_batter_pos_y_inches',
                      'n_priorpa_thisgame_player_at_bat', 'n_thruorder_pitcher',
                      'release_pos_x', 'release_pos_z', 'events', 'description', 'spin_dir', 'spin_rate_deprecated',
                      'zone', 'des', 'game_type', 'stand', 'p_throws', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
                      'inning', 'hc_x', 'hc_y', 'tfs_deprecated', 'tfs_zulu_deprecated', 'sv_id', 'vx0', 'vy0',
                      'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'release_pos_y', 'iso_value',
                      'pitcher_days_until_next_game', 'api_break_z_with_gravity', 'api_break_x_arm',
                      'api_break_x_batter_in', 'arm_angle', 'swing_path_tilt', 'type', 'release_extension', ], axis=1)

# print(myStatsData.columns)

myStatsData.to_csv("2020pitchlog.csv")