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
