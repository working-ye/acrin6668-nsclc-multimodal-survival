# Security and privacy reporting

Do not open a public issue containing patient information, credentials, private
paths, model bundles, predictions, or non-public study results. Use GitHub's
private vulnerability-reporting feature or contact the repository maintainers
through an approved institutional channel.

Only load `.joblib` artifacts produced by a trusted run of this code. Python
pickle-based formats can execute arbitrary code during deserialization.
