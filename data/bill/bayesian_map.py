'''
Method 1: Bayesian burn mapping.

This method features Bayesian belief update. 

For current method, it uses Beta-Binomial conjugate.

Current version: throws away spatial information (all pixels are independent).

Assumption: A no signal needs at least N days (e.g 3 images) to start being burn.
'''

from barc import (
    dNBR,
    dnbr_256
)



def is_evidence(
        dnbr    
):
    '''
    returns booleans.
    '''
    
    medium_severity = dnbr_256(dnbr, threshold=78)

    return medium_severity



def beta_expectation(
        alpha, beta
):
    '''
    Returns the mean of beta distribution.
    '''

    return alpha / (alpha + beta + 1e-3)



def make_prediction(
        expectations, 
        threshold: float
):
    
    return expectations >= threshold



def bayesian_update_1(
        *,
        alpha,
        beta,
        new_dnbr
):
    '''
    3-days method
    '''
    
    evidence = is_evidence(new_dnbr)

    new_alpha = alpha + evidence
    new_beta  = beta + ~evidence

    #Update feature
    mask = (new_alpha + 3) / (new_alpha + new_beta + 3) < 0.5#Only no evidence can 
    new_beta = new_beta - (mask) * (~evidence)

    return new_alpha, new_beta



def bayesian_update_2(
        *,
        alpha,
        beta,
        new_dnbr
):
    '''

    '''
    
    evidence = is_evidence(new_dnbr)

    new_alpha = alpha + evidence
    new_beta  = beta + ~evidence

    #Update feature
    mask = (new_alpha + 4) / (new_alpha + new_beta + 4) < 0.5#Only no evidence can 
    new_beta = new_beta - (mask) * (~evidence)

    return new_alpha, new_beta

    

