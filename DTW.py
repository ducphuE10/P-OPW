from dtaidistance import dtw


def dtw_distance(X,Y):
    return dtw.distance_fast(X, Y, use_pruning=True)