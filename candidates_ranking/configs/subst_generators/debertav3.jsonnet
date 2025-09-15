local prob_estimator = import '../prob_estimators/debertav3.jsonnet';

{
    class_name: "SubstituteGenerator",
    prob_estimator: prob_estimator,
}
