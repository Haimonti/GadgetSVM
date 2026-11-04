from . import centralized, fedavg_svm, bdsvm, fdr_svm, fedssl_amc

METHODS = {
    "centralized": centralized,
    "fedavg_svm":  fedavg_svm,
    "bdsvm":       bdsvm,
    "fdr_svm":     fdr_svm,
    "fedssl_amc":  fedssl_amc,
}
