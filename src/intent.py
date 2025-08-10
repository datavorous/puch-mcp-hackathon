from math import log, sqrt, exp


def get_intent(
    upvotes,
    comments,
    ratio,
    created_utc,
    current_utc,
    author_karma=1,
    keyword_matches=0,
) -> float:

    try:
        age_hours = max((current_utc - created_utc) / 3600, 0.01)
        engagement_velocity = (upvotes + comments * 3) / age_hours
        velocity_score = log(1 + engagement_velocity) / log(1000)

        n = upvotes / max(ratio, 0.01)
        if n > 0:
            z = 1.96
            phat = upvotes / n
            wilson = (
                phat
                + z * z / (2 * n)
                - z * sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
            ) / (1 + z * z / n)
            wilson_normalized = max(wilson, 0)
        else:
            wilson_normalized = 0

        snr_score = min(comments / sqrt(upvotes) / 10, 1.0) if upvotes > 0 else 0
        authority = log(1 + author_karma) / log(1 + 1000000)
        keyword_saturation = 1 - exp(-keyword_matches * 0.5)

        components = [velocity_score, wilson_normalized, snr_score, keyword_saturation]
        if all(x > 0 for x in components):
            geometric_mean = pow(
                velocity_score * wilson_normalized * snr_score * keyword_saturation,
                1 / 4,
            )
        else:
            geometric_mean = 0

        intent_score = round(
            (
                100
                / (
                    1
                    + exp(-0.1 * (geometric_mean * (0.7 + 0.3 * authority) * 100 - 50))
                )
            ),
            2,
        )

        return intent_score

    except Exception as e:
        print(f"Error calculating intent score: {str(e)}")
        return 0.0
