import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X
from copy import deepcopy

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # TODO: Perform the optimization and print the result
    # Running the optimization
    result = optimizer.optimize()

    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    

    marginals_list = []
    land_mark_options = [1, 2]
    
    for landmark in land_mark_options:
        for key in pose_options:
            # Fresh copy for each combination
            graph_copy = deepcopy(graph)
            estimate_copy = gtsam.Values(initial_estimate)
            
            pose_5 = pose_options[key]
            
            # Use copies, not originals
            graph_copy, estimate_copy = add_pose(graph_copy, estimate_copy, pose_5)
            result = optimize(graph_copy, estimate_copy)
            
            graph_copy = add_landmark_measurement(graph_copy, result, pose_5, landmark)
            result = optimize(graph_copy, estimate_copy)
            
            marginals = gtsam.Marginals(graph_copy, result)
            cov = marginals.marginalCovariance(L(landmark))

            print(f"pose: {key}, landmark: {landmark}")
            print(f"covariance matrix:\n{cov}")
            print(f"sum: {cov.sum()}")
            print(f"trace: {np.trace(cov)}")
            print(f"det: {np.linalg.det(cov)}")
            print("---")

            marginals_list.append(np.trace(cov))

            cov_x4 = marginals.marginalCovariance(X(4))
            cov_l1 = marginals.marginalCovariance(L(1))
            cov_l2 = marginals.marginalCovariance(L(2))

            print(f"trace X4: {np.trace(cov_x4)}")
            print(f"trace L1: {np.trace(cov_l1)}")
            print(f"trace L2: {np.trace(cov_l2)}")
            print(f"sum of all traces: {np.trace(cov_x4) + np.trace(cov_l1) + np.trace(cov_l2)}")
            print(f"sum L1+L2: {np.trace(cov_l1) + np.trace(cov_l2)}")

    print(f"all marginals: {marginals_list}")
    print(f"target value: 0.15697937979143664")
    
    index_lowest_marginals = np.argmin(marginals_list)
    if index_lowest_marginals <= 3:
        best_pose = list(pose_options)[index_lowest_marginals]
        best_landmark = 1 
    else:
        best_pose = list(pose_options)[index_lowest_marginals - 4]
        best_landmark = 2

    # Return the actual marginals object for the best combination
    sum_of_marginals = marginals_list[index_lowest_marginals]

    #print(f"pose is {best_pose} and landmark is {best_landmark}")
    return best_pose, best_landmark, sum_of_marginals


    # marginals_list = []

    # land_mark_options = [1, 2]
    # for landmark in land_mark_options:

    #     for key in pose_options:

    #         graph_copy = deepcopy(graph)
    #         estimate_copy = gtsam.Values(initial_estimate)

    #         pose_5 = pose_options[key]
    #         graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    #         result = optimize(graph, initial_estimate)
    #         graph = add_landmark_measurement(graph, result, pose_5, landmark)
    #         result = optimize(graph, initial_estimate)

    #         marginals = gtsam.Marginals(graph, result) # calculates the marginals for all poses adn landmarks 
    #         marginal_sum = marginals.marginalCovariance(L(landmark)).sum() # calculates the sum of marginals form the currct landmark 

    #         marginals_list.append(marginal_sum)
    
    # index_lowest_marginals = np.argmin(marginals_list)

    # if index_lowest_marginals <= 3:
    #     best_pose = list(pose_options)[index_lowest_marginals]
    #     best_landmark = 1 

    # else:
    #     best_pose = list(pose_options)[index_lowest_marginals - 4] 
    #     best_landmark = 2


    # # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
    # # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
    # sum_of_marginals = marginals_list[index_lowest_marginals]


    # return best_pose, best_landmark, sum_of_marginals










def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = "a"      # chosen pose option
    best_landmark = 1    # chosen landmark (1 or 2)
    pose_5 = pose_options[best_pose]
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result = optimize(graph, initial_estimate)
    graph = add_landmark_measurement(graph, result, pose_5, best_landmark)
    result = optimize(graph, initial_estimate)

    # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
    list_of_errors = []
    # TODO: compute the sum of the errors and return it along with the best pose and landmark
    sum_of_errors = 0
    return best_pose, best_landmark, sum_of_errors 