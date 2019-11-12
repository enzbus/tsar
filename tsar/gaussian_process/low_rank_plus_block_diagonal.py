from collections import namedtuple

low_rank_plus_block_diagonal = \
    namedtuple('D', 'Sigma_lr', 'V')


def fit_low_rank_block_diagonal(lagged_covs, Sigma_hat, R, P, F):
    # lagged_covs = make_lagged_covariances(normalized_residual, lag=P+F)
    # Sigma_hat = build_dense_covariance_matrix(lagged_covs)
    v = compute_principal_directions(R, lagged_covs)
    V = make_V(v, lag=P+F)
    logger.info('Computing Sigma_lr')
    Sigma_lr = V @ Sigma_hat @ V.T
    logger.info('Computing D.')
    blocks = []
    for m in range(lagged_covs.shape[0]):
        only_low_rank = V[:, m*(P+F):(m+1)*(P+F)].T @ Sigma_lr \
            @ V[:, m*(P+F):(m+1)*(P+F)]
        original = Sigma_hat[m*(P+F):(m+1)*(P+F), m*(P+F):(m+1)*(P+F)]
        blocks.append(original-only_low_rank)
    D = sp.bmat(blocks)
    return Sigma_lr, V, D


def woodbury_inverse(V: np.matrix,  # sp.csc.csc_matrix,
                     S_inv: np.matrix,
                     D_inv: np.matrix):
    """ https://en.wikipedia.org/wiki/Woodbury_matrix_identity

    Compute (V @ S @ V.T + D)^-1.
    """
    #assert V.__class__ is sp.csc.csc_matrix
    assert (S_inv.__class__ is np.matrix) or (S_inv.__class__ is np.ndarray)
    assert D_inv.__class__ is np.matrix

    #V = V.todense()

    logger.debug('Solving Woodbury inverse.')
    logger.debug('Building internal matrix.')
    internal = S_inv + V.T @ D_inv @ V
    logger.debug('Inverting internal matrix.')
    intinv = np.linalg.inv(
        internal.todense() if hasattr(
            internal,
            'todense') else internal)
    logger.debug('Building inverse.')
    # D_invV = (D_inv @ V)
    # return D_inv - D_invV @ intinv @ D_invV.T

    return D_inv - (D_inv @ (V @ intinv)) @ (D_inv @ V).T
