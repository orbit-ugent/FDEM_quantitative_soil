def stochastic_poly_layer(df, feature_columns, target, layer_cm, n=4, iters=100, round_n=3):
    ypred_train_best, ypred_test_best, R2_train_t_best, R2_test_t_best, RMSE_train_t_best, RMSE_test_t_best = [], [], [], [], [], []

    for i in range(iters):
        X_layer_cm = df[df['depth'] == layer_cm][feature_columns[0]].values.reshape(-1, 1)
        Y_layer_cm = df[df['depth'] == layer_cm][target].values

        X_train, X_test, y_train, y_test = train_test_split(X_layer_cm, Y_layer_cm, test_size=0.3, random_state=i)
        LinReg = LinearRegression()
        ypred_train_, ypred_test_, R2_train_t_, R2_test_t_, RMSE_train_t_, RMSE_test_t_ = [], [], [], [], [], []

        for k in range(n):
            poly = PolynomialFeatures(degree=k)
            poly.fit(X_train)
            Xt_train = poly.transform(X_train)
            Xt_test = poly.transform(X_test)

            LinReg.fit(Xt_train, y_train)
            ypred_train = LinReg.predict(Xt_train)
            ypred_test = LinReg.predict(Xt_test)

            R2_train_t = r2_score(y_train, ypred_train)
            R2_test_t = r2_score(y_test, ypred_test)
            RMSE_train_t = RMSE(y_train, ypred_train)
            RMSE_test_t = RMSE(y_test, ypred_test)

            ypred_train_.append(ypred_train)
            ypred_test_.append(ypred_test)
            R2_train_t_.append(R2_train_t)
            R2_test_t_.append(R2_test_t)
            RMSE_train_t_.append(RMSE_train_t)
            RMSE_test_t_.append(RMSE_test_t)

        ypred_train_best.append(ypred_train_)
        ypred_test_best.append(ypred_test_)
        R2_train_t_best.append(R2_train_t_)
        R2_test_t_best.append(R2_test_t_)
        RMSE_train_t_best.append(RMSE_train_t_)
        RMSE_test_t_best.append(RMSE_test_t_)

    RMSE_test_n1 = [inner_list[0] for inner_list in RMSE_test_t_best]
    RMSE_test_n2 = [inner_list[1] for inner_list in RMSE_test_t_best]
    RMSE_test_n3 = [inner_list[2] for inner_list in RMSE_test_t_best]
    RMSE_test_n4 = [inner_list[3] for inner_list in RMSE_test_t_best]

    RMSE_sums = [np.sum(RMSE_test_n1), np.sum(RMSE_test_n2), np.sum(RMSE_test_n3), np.sum(RMSE_test_n4)]
    best_n = RMSE_sums.index(np.min(RMSE_sums))
    return best_n, round(np.mean([inner_list[best_n] for inner_list in R2_test_t_best]), round_n), round(np.mean([inner_list[best_n] for inner_list in R2_train_t_best]), round_n), round(np.mean([inner_list[best_n] for inner_list in RMSE_test_t_best]), round_n), round(np.mean([inner_list[best_n] for inner_list in RMSE_train_t_best]), round_n)

######################################################################################################################################################

target_set = [
    'vwc',
    'CEC',
    'clay',
    'bd',
    'water_ec_hp'
]

for depth_ in [10, 50]:
    i = 0

    for t in target_set:
        R2_test, R2_train, n_ = [], [], []
        target = ds_all[t].values

        for feature_set in feature_sets:        
            best_n, R2_test_pol, R2_train_pol, RMSE_test_pol, RMSE_train_pol = stochastic_poly_layer(ds_all, feature_set, t, depth_, iters=100)
            R2_test.append(R2_test_pol)
            R2_train.append(R2_train_pol)
            n_.append(best_n)

        best_index = R2_test.index(np.max(R2_test))
        n = n_[best_index]
        R2_stochastic['Best EC feature '+str(depth_)+'cm'][i] = feature_sets[best_index]
        R2_stochastic['Target'][i] = t
        R2_stochastic['R2 '+str(depth_)+'cm'][i] = R2_test[best_index]
        print('for predicting stochastically '+t+' the best predictor is: ' +feature_sets[best_index][0]+' with ply grade: '+str(n))
        bars_plot(feature_sets, R2_test, R2_train, t)
        implementation(ds_all[ds_all['depth'] == depth_], feature_sets[best_index], t, n)
        i+=1

R2_stochastic

######################################################################################################################################################

# Initialize DataFrames with one row for the single target
R2_deterministic = pd.DataFrame(columns=['Target', 'Best EC feature LT', 'R2 LT', 'Best EC feature ID', 'R2 ID', 'Best EC feature LS', 'R2 LS'],
                                index=[0])

RMSE_deterministic = pd.DataFrame(columns=['Target', 'Best EC feature LT', 'RMSE LT', 'Best EC feature ID', 'RMSE ID', 'Best EC feature LS', 'RMSE LS'],
                                  index=[0])

error_criteria_selection = 'RMSE'

# Initialize variables to store best features and their scores
median_RMSE_LT_, median_RMSE_LS_, median_RMSE_ID_ = [], [], []
median_R2_LT_, median_R2_LS_, median_R2_ID_ = [], [], []

# Iterate over features to find the best one
for feature, scores in results.items():
    median_R2_LT = np.median(scores['LT']['R2'])
    median_R2_LS = np.median(scores['LS']['R2'])
    median_R2_ID = np.median(scores['ID']['R2'])

    median_RMSE_LT = np.median(scores['LT']['RMSE'])
    median_RMSE_LS = np.median(scores['LS']['RMSE'])
    median_RMSE_ID = np.median(scores['ID']['RMSE'])

    median_RMSE_LT_.append(median_RMSE_LT)
    median_RMSE_LS_.append(median_RMSE_LS)
    median_RMSE_ID_.append(median_RMSE_ID)

    median_R2_LT_.append(median_R2_LT)
    median_R2_LS_.append(median_R2_LS)
    median_R2_ID_.append(median_R2_ID)

    if error_criteria_selection == 'RMSE':
        best_feature_LT = median_RMSE_LT_.index(np.min(median_RMSE_LT_))
        best_feature_LS = median_RMSE_LS_.index(np.min(median_RMSE_LS_))
        best_feature_ID = median_RMSE_ID_.index(np.min(median_RMSE_ID_))

    if error_criteria_selection == 'R2':
        best_feature_LT = median_R2_LT_.index(np.min(median_R2_LT_))
        best_feature_LS = median_R2_LS_.index(np.min(median_R2_LS_))
        best_feature_ID = median_R2_ID_.index(np.min(median_R2_ID_))


# Update DataFrames
R2_deterministic.loc[0] = [target, best_feature_LT, max_r2_lt, best_feature_r2_ls, max_r2_ls, best_feature_r2_ideal, max_r2_ideal]
RMSE_deterministic.loc[0] = [target, best_feature_LT, min_rmse_lt, best_feature_rmse_ls, min_rmse_ls, best_feature_rmse_ideal, min_rmse_ideal]
#################################################################################

error_criteria_selection='R2'

    if error_criteria_selection == 'RMSE':

        RMSE_median_10 = [np.median(RMSE_test_n1_10), np.median(RMSE_test_n2_10), np.median(RMSE_test_n3_10), np.median(RMSE_test_n4_10)]
        best_n_10 = RMSE_median_10.index(np.min(RMSE_median_10))

        RMSE_median_50 = [np.median(RMSE_test_n1_50), np.median(RMSE_test_n2_50), np.median(RMSE_test_n3_50), np.median(RMSE_test_n4_50)]
        best_n_50 = RMSE_median_50.index(np.min(RMSE_median_50))

        RMSE_median_LS = [np.median(RMSE_test_n1_LS), np.median(RMSE_test_n2_LS), np.median(RMSE_test_n3_LS), np.median(RMSE_test_n4_LS)]
        best_n = RMSE_median_LS.index(np.min(RMSE_median_LS))

        RMSE_median_LT = [np.median(RMSE_test_n1_LT), np.median(RMSE_test_n2_LT), np.median(RMSE_test_n3_LT), np.median(RMSE_test_n4_LT)]
        best_n_LT = RMSE_median_LT.index(np.min(RMSE_median_LT))

    elif error_criteria_selection == 'R2':

        R2_median_10 = [np.median(R2_test_n1_10), np.median(R2_test_n2_10), np.median(R2_test_n3_10), np.median(R2_test_n4_10)]
        best_n_10 = R2_median_10.index(np.max(R2_median_10))

        R2_median_50 = [np.median(R2_test_n1_50), np.median(R2_test_n2_50), np.median(R2_test_n3_50), np.median(R2_test_n4_50)]
        best_n_50 = R2_median_50.index(np.max(R2_median_50))

        R2_median = [np.median(R2_test_n1_LS), np.median(R2_test_n2_LS), np.median(R2_test_n3_LS), np.median(R2_test_n4_LS)]
        best_n = R2_median.index(np.max(R2_median))

        R2_median_LT = [np.median(R2_test_n1_LT), np.median(R2_test_n2_LT), np.median(R2_test_n3_LT), np.median(R2_test_n4_LT)]
        best_n_LT = R2_median_LT.index(np.max(R2_median_LT))

##########
        R2_test_n0_10 = [inner_list[0] for inner_list in SR2_test_10_]
        R2_test_n1_10 = [inner_list[1] for inner_list in SR2_test_10_]
        R2_test_n2_10 = [inner_list[2] for inner_list in SR2_test_10_]
        R2_test_n3_10 = [inner_list[3] for inner_list in SR2_test_10_]

        R2_test_n0_50 = [inner_list[0] for inner_list in SR2_test_50_]
        R2_test_n1_50 = [inner_list[1] for inner_list in SR2_test_50_]
        R2_test_n2_50 = [inner_list[2] for inner_list in SR2_test_50_]
        R2_test_n3_50 = [inner_list[3] for inner_list in SR2_test_50_]


        RMSE_test_n0_10 = [inner_list[0] for inner_list in SRMSE_test_10_]
        RMSE_test_n1_10 = [inner_list[1] for inner_list in SRMSE_test_10_]
        RMSE_test_n2_10 = [inner_list[2] for inner_list in SRMSE_test_10_]
        RMSE_test_n3_10 = [inner_list[3] for inner_list in SRMSE_test_10_]

        RMSE_test_n0_50 = [inner_list[0] for inner_list in SRMSE_test_50_]
        RMSE_test_n1_50 = [inner_list[1] for inner_list in SRMSE_test_50_]
        RMSE_test_n2_50 = [inner_list[2] for inner_list in SRMSE_test_50_]
        RMSE_test_n3_50 = [inner_list[3] for inner_list in SRMSE_test_50_]

##################

        if criteria == 'RMSE':

            SRMSE_LT = [SRMSE_test_LT0, SRMSE_test_LT1, SRMSE_test_LT2, SRMSE_test_LT3]
            SRMSE_meadian_LT = [np.median(sublist) for sublist in SRMSE_LT]
            best_n_LT = SRMSE_meadian_LT.index(np.max(SRMSE_meadian_LT))
            SRMSE_test_LTb = SRMSE_LT[best_n_LT]
        
            SR2_LT = [SR2_test_LT0, SR2_test_LT1, SR2_test_LT2, SR2_test_LT3]
            SR2_meadian_LT = [np.median(sublist) for sublist in SR2_LT]
            SR2_test_LTb = SR2_LT[best_n_LT]

            SRMSE_LS = [SRMSE_test_LS0, SRMSE_test_LS1, SRMSE_test_LS2, SRMSE_test_LS3]
            SRMSE_meadian_LS = [np.median(sublist) for sublist in SRMSE_LS]
            best_n_LS = SRMSE_meadian_LS.index(np.max(SRMSE_meadian_LS))
            SRMSE_test_LSb = SRMSE_LS[best_n_LS]

            SR2_LS = [SR2_test_LS0, SR2_test_LS1, SR2_test_LS2, SR2_test_LS3]
            SR2_meadian_LS = [np.median(sublist) for sublist in SR2_LS]
            SR2_test_LSb = SR2_LS[best_n_LS]

            SRMSE_LS2 = SR2_test_LS2b
            SRMSE_meadian_LS2 = [np.median(sublist) for sublist in SRMSE_LS2]
            best_n_LS2 = SRMSE_meadian_LS2.index(np.max(SRMSE_meadian_LS2))
            SRMSE_test_LS2b = SRMSE_LS2[best_n_LS2]

            SR2_LS2 = SRMSE_test_LS2b
            SR2_meadian_LS2 = [np.median(sublist) for sublist in SR2_LS2]
            SR2_test_LS2b = SR2_LS2[best_n_LS2]
