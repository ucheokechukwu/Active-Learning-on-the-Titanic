def class_balancing(df_, target_class, reduce_larger_class=False):
    """Takes df, checks for class imbalances
    returns X and y with proper balancing
    Only works for binary classes at the moment.
    """

    if reduce_larger_class:
        Print ("Does not have this function yet")
        return

    df = df_.copy()
    # extend the smaller class
    y = df[target_class]
    display(y.value_counts())

    # find the larger value

    imbalance = np.abs(y.value_counts()[0] - y.value_counts()[1])
    smaller_class = np.argmin(y.value_counts()) 

    print(f"We need {imbalance} new {smaller_class} records to offset class imbalance.")


    filter = y == smaller_class
    tmp = df[filter].sample(n=imbalance, replace=True)

    df_index = df.index.to_list() + tmp.index.to_list()
    np.random.shuffle(df_index) # shuffling is important
    df = df.loc[df_index].copy()
    # checking the value counts
    X = df.drop(columns=target_class)
    y = df[target_class]
    display(y.value_counts())

    # reseting the index to handle the duplicated records.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y
