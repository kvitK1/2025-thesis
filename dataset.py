import polars as pl
from datetime import datetime
import re


def load_data(path_to_folder: str) -> pl.DataFrame:
    df = pl.read_parquet(path_to_folder + "/*.parquet")
    return df


def process_data(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(pl.col("payment_view_time").is_not_null())

    df = df.with_columns(
        [
            pl.col("journey_name_congrats_view_time")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            .alias("start_time"),
            pl.col("payment_view_time")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            .alias("end_time"),
        ]
    )

    df = df.filter(pl.col("end_time") > pl.col("start_time"))

    selected_columns = [
        "purchase_date",
        "product_id",
        "start_time",
        "end_time",
        "device_type",
        "country",
        "language",
        "version",
        "media_source_type",
        "media_source",
        "what_summary_is_answer",
        "survey_answer",
        "social_proof_type",
        "statement_1",
        "statement_2",
        "statement_3",
        "goal_adjust_goal",
        "question_0",
        "question_1",
        "question_2",
        "push_allow",
        "time_periods",
        "life_desires",
        "gender",
        "age",
        "streak_days",
        "time_periods_count",
        "life_desires_count",
        "selected_books_count",
        "payment_view_count",
        "commitment_identity",
        "commitment_goal",
        "discover_for_you_view_count",
        "subscription_intent_count",
        "onboarding_summary_scrolled_to_bottom_count",
    ]
    df = df.select(selected_columns)

    columns_of_interest = [
        "device_type",
        "country",
        "language",
        "version",
        "media_source_type",
        "media_source",
        "what_summary_is_answer",
        "survey_answer",
        "social_proof_type",
        "statement_1",
        "statement_2",
        "statement_3",
        "goal_adjust_goal",
        "question_0",
        "question_1",
        "question_2",
        "push_allow",
        "time_periods",
        "life_desires",
        "gender",
        "age",
        "streak_days",
        "time_periods_count",
        "life_desires_count",
        "selected_books_count",
        "payment_view_count",
        "commitment_identity",
        "commitment_goal",
        "discover_for_you_view_count",
        "subscription_intent_count",
        "onboarding_summary_scrolled_to_bottom_count",
    ]

    for column in columns_of_interest:
        total = df[column].len()

        value_counts = df[column].value_counts().sort("count", descending=True)

        value_counts = value_counts.with_columns(
            [((pl.col("count") / total) * 100).round(2).alias("% share")]
        )
        print(column)
        print(value_counts)

    return df


def feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    categorical_cols = [
        "device_type",
        "country",
        "language",
        "media_source_type",
        "media_source",
        "gender",
        "age",
        "what_summary_is_answer",
        "survey_answer",
        "social_proof_type",
        "statement_1",
        "statement_2",
        "statement_3",
        "question_0",
        "question_1",
        "question_2",
        "push_allow",
        "time_periods",
        "life_desires",
    ]

    df = df.with_columns(
        [pl.col(col).fill_null("missing").alias(col) for col in categorical_cols]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("version").is_not_null())
            .then(pl.col("version").str.split(".").list.get(0).cast(pl.Float64))
            .otherwise(None)
            .alias("version_float")
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("version_float") >= 299)
            .then(pl.lit("after_299"))
            .when(pl.col("version_float").is_not_null())
            .then(pl.lit("before_299"))
            .otherwise(pl.lit("missing"))
            .alias("version_grouped")
        ]
    )

    # BAU region distribution
    europe = [
        "Albania",
        "Andorra",
        "Austria",
        "Belgium",
        "Bosnia and Herzegovina",
        "Bulgaria",
        "Croatia",
        "Cyprus",
        "Czech Republic",
        "Denmark",
        "Estonia",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Hungary",
        "Ireland",
        "Italy",
        "Latvia",
        "Liechtenstein",
        "Lithuania",
        "Luxembourg",
        "Macedonia",
        "Malta",
        "Moldova",
        "Monaco",
        "Netherlands",
        "Norway",
        "Poland",
        "Portugal",
        "Romania",
        "Serbia",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Sweden",
        "Switzerland",
    ]

    latam = [
        "Argentina",
        "Bolivia",
        "Brazil",
        "Chile",
        "Colombia",
        "Costa Rica",
        "Dominican Republic",
        "Ecuador",
        "El Salvador",
        "Equatorial Guinea",
        "Guatemala",
        "Honduras",
        "Mexico",
        "Nicaragua",
        "Panama",
        "Paraguay",
        "Peru",
        "Puerto Rico",
        "Uruguay",
        "Venezuela",
    ]

    df = df.with_columns(
        [
            pl.when(pl.col("country") == "United States")
            .then(pl.lit("United States"))
            .when(pl.col("country") == "Canada")
            .then(pl.lit("Canada"))
            .when(pl.col("country") == "United Kingdom")
            .then(pl.lit("United Kingdom"))
            .when(pl.col("country") == "Australia")
            .then(pl.lit("Australia"))
            .when(pl.col("country").is_in(europe))
            .then(pl.lit("Europe"))
            .when(pl.col("country").is_in(latam))
            .then(pl.lit("Latam"))
            .otherwise(pl.lit("Other"))
            .alias("country_grouped")
        ]
    )

    translation_map = {
        # French
        "Avant de m'endormir": "Before going to sleep",
        "Pendant le temps libre": "Any spare time",
        "Pendant ma pause déjeuner": "During my lunch break",
        "Pendant les trajets domicile-travail": "While commuting",
        "Après le café du matin": "After morning coffee",
        # Spanish
        "Antes de ir a dormir": "Before going to sleep",
        "Cualquier momento libre": "Any spare time",
        "Durante mi pausa para el almuerzo": "During my lunch break",
        "Mientras me desplazo": "While commuting",
        "Después del café de la mañana": "After morning coffee",
        # German
        "Vor dem Einschlafen": "Before going to sleep",
        "Jederzeit": "Any spare time",
        "In meiner Mittagspause": "During my lunch break",
        "Auf dem Weg zur Arbeit": "While commuting",
        "Nach dem Frühstück": "After morning coffee",
        "Nach dem Morgenkaffee": "After morning coffee",
        # English already standardized
        "Before going to sleep": "Before going to sleep",
        "Any spare time": "Any spare time",
        "During my lunch break": "During my lunch break",
        "While commuting": "While commuting",
        "After morning coffee": "After morning coffee",
    }

    def normalize_summary_answer(answer):
        if answer is None:
            return None

        items = [i.strip() for i in answer.split(",")]

        translated = []
        for i in items:
            if i in translation_map:
                translated.append(translation_map[i])
            else:
                return "Other"

        return ",".join(sorted(set(translated)))

    df = df.with_columns(
        [
            pl.col("time_periods")
            .map_elements(
                normalize_summary_answer, skip_nulls=False, return_dtype=pl.Utf8
            )
            .alias("time_periods_eng")
        ]
    )

    allowed_values = ["18", "25", "35", "45", "55", "missing"]
    pattern = "|".join(allowed_values)

    df = df.with_columns(
        pl.when(pl.col("age").str.contains(pattern))
        .then(pl.col("age"))
        .otherwise(pl.lit("Other"))
        .alias("age_grouped")
    )

    def categorize_device(model):
        if model is None:
            return "missing"

        model = model.lower()

        if "iphone" in model:
            if any(x in model for x in ["12", "13", "14", "15", "16"]):
                return "iPhone_new"
            else:
                return "iPhone_old"

        elif "ipad" in model:
            return "iPad"

        else:
            return "Other"

    df = df.with_columns(
        [
            pl.col("device_type")
            .map_elements(categorize_device, return_dtype=pl.Utf8)
            .alias("device_grouped")
        ]
    )

    allowed_values = ["female", "male", "missing"]

    pattern = "|".join(allowed_values)

    df = df.with_columns(
        pl.when(pl.col("gender").str.contains(pattern))
        .then(pl.col("gender"))
        .otherwise(pl.lit("Other"))
        .alias("gender_grouped")
    )

    # Latam users have a little bit shorter onboarding, so they don't have statement_3 step, we'll send them to new category
    df = df.with_columns(
        pl.when(pl.col("country_grouped") == "Latam")
        .then(pl.lit("no_step"))
        .otherwise(pl.col("statement_3"))
        .alias("statement_3")
    )

    df = df.with_columns(
        pl.when(pl.col("purchase_date").is_null())
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("purchase")
    )

    def group_product_id(product_id: str) -> str:
        """Groups a product_id based on predefined patterns."""

        if product_id is None:
            return "missing"

        product_id = product_id.strip().lower()

        if re.search(r"trial_yearly", product_id):
            return "trial_yearly"
        elif re.search(r"annually|yearly", product_id) and re.search(r"pro", product_id):
            return "paid_yearly_pro"
        elif re.search(r"annually|yearly", product_id):
            return "paid_yearly"
        elif re.search(r"monthly", product_id) and re.search(r"pro", product_id):
            return "paid_monthly_pro"
        elif re.search(r"monthly", product_id):
            return "paid_monthly"
        elif re.search(r"weekly", product_id):
            return "paid_weekly"
        elif re.search(r"3months", product_id):
            return "paid_3months"
        elif product_id == "":
            return "missing"
        else:
            return "other"
    
    df = df.with_columns(
        pl.col("product_id")
        .map_elements(group_product_id, return_dtype=pl.Utf8)
        .alias("product_id_grouped")
    )

    return df


def numerical_to_categorical(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            ((pl.col("end_time") - pl.col("start_time")).cast(pl.Int64) / 1e6).alias(
                "quiz_duration"
            )
        ]
    )

    df = df.filter(pl.col("quiz_duration") < 1800)

    def quiz_duration_group(duration):
        if duration is None:
            return "missing"
        if duration <= 250:
            return "short"
        elif duration <= 400:
            return "medium"
        else:
            return "long"

    df = df.with_columns(
        [
            pl.col("quiz_duration")
            .map_elements(quiz_duration_group, return_dtype=pl.Utf8)
            .alias("quiz_duration_grouped")
        ]
    )

    df = df.with_columns(
        [
            pl.col("goal_adjust_goal")
            .fill_null(0)
            .cast(pl.Int64)
            .alias("goal_adjust_goal")
        ]
    )

    df = df.filter(pl.col("goal_adjust_goal") < 50)

    df = df.with_columns(
        [
            pl.when(pl.col("goal_adjust_goal") < 5)
            .then(pl.lit("small_goal"))
            .when(pl.col("goal_adjust_goal") < 15)
            .then(pl.lit("medium_goal"))
            .when(pl.col("goal_adjust_goal") == 15)
            .then(pl.lit("default_goal"))
            .otherwise(pl.lit("big_goal"))
            .alias("goal_adjust_grouped")
        ]
    )

    df = df.with_columns(
        [pl.col("streak_days").fill_null(0).cast(pl.Int64).alias("streak_days")]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("streak_days") == 0)
            .then(pl.lit("no_commit"))
            .when(pl.col("streak_days") == 7)
            .then(pl.lit("basic_commit"))
            .otherwise(pl.lit("strong_commit"))
            .alias("streak_commitment_level")
        ]
    )

    df = df.with_columns(
        [
            pl.col("time_periods_count")
            .fill_null(0)
            .cast(pl.Int64)
            .alias("time_periods_count")
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("time_periods_count") == 0).then(pl.lit("none"))
            .when(pl.col("time_periods_count") == 1).then(pl.lit("focused"))
            .otherwise(pl.lit("flexible"))
            .alias("engagement_window")
        ]
    )

    df = df.with_columns(
        [
            pl.col("life_desires_count")
            .fill_null(0)
            .cast(pl.Int64)
            .alias("life_desires_count")
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("life_desires_count") == 0)
            .then(pl.lit("missing"))
            .when(pl.col("life_desires_count") == 3)
            .then(pl.lit("expected"))
            .otherwise(pl.lit("unusual"))
            .alias("life_desires_engagement")
        ]
    )

    df = df.with_columns(
        [
            pl.col("selected_books_count")
            .fill_null(0)
            .cast(pl.Int64)
            .alias("selected_books_count")
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("selected_books_count") == 0)
            .then(pl.lit("missing"))
            .when(pl.col("selected_books_count") == 5)
            .then(pl.lit("expected"))
            .otherwise(pl.lit("unusual"))
            .alias("selected_books_grouped")
        ]
    )

    df = df.with_columns(
        [
            pl.when(
                (
                    ((pl.col("start_time").dt.month() == 7) & (pl.col("start_time").dt.day() >= 16)) |
                    ((pl.col("start_time").dt.month() == 8) & (pl.col("start_time").dt.day() <= 30)) |
                    ((pl.col("start_time").dt.month() == 12) & (pl.col("start_time").dt.day() >= 22)) |
                    ((pl.col("start_time").dt.month() == 1) & (pl.col("start_time").dt.day() <= 31)) |
                    ((pl.col("start_time").dt.month() == 2) & (pl.col("start_time").dt.day() == 1))
                )
            )
            .then(pl.lit("High season"))
            .otherwise(pl.lit("Not high"))
            .alias("season_category")
        ]
    )

    return df


def main():
    start = datetime.now()

    path_to_folder = "data/onboarding_parquets"
    df = load_data(path_to_folder)
    df = process_data(df)
    df = feature_engineering(df)
    df = numerical_to_categorical(df)

    output_path = "data/processed_onboarding.parquet"
    df.write_parquet(output_path)

    print(f"Processed DataFrame saved to {output_path}")

    print(f"Time taken to process files: {datetime.now() - start}")

if __name__ == "__main__":
    main()
