import pyten.UI


def helios(scenario=None):
    """
    Helios Main API returns decomposition or Recovery Result of All three scenario
    """

    # Initialization
    Ori = None  # Original Tensor
    full = None  # Full Tensor reconstructed by decomposed matrices
    Final = None  # Decomposition Results e.g. Ttensor or Ktensor
    Rec = None  # Recovered Tensor (Completed Tensor)

    # User Interface
    if scenario is None:
        scenario = raw_input("Please choose the scenario:\n"
                             " 1. Basic Tensor Decomposition/Completion  2.Tensor Decompostion/Completion with Auxiliary Information"
                             " 3.Dynamic Tensor Decomposition/Completion 4.Scalable Tensor Decomposition/Completion 0.Exit \n")

    if scenario == '1':  # Basic Tensor Decomposition/Completion
        [Ori, full, Final, Rec] = pyten.UI.basic()
    elif scenario == '2':  # Tensor Completion with Auxiliary Information
        [Ori, full, Final, Rec] = pyten.UI.auxiliary()
    elif scenario == '3':  # Dynamic Tensor Decomposition
        [Ori, full, Final, Rec] = pyten.UI.dynamic()
    elif scenario == '4':  # Dynamic Tensor Decomposition
        [Ori, full, Final, Rec] = pyten.UI.scalable()
    elif scenario == '0':
        print 'Successfully Exit'
        return Ori, full, Final, Rec
    else:
        raise ValueError('No Such scenario')

    # Return result
    return Ori, full, Final, Rec
