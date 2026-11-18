
from analyzer import (
    load_expenses,
    analyze_expenses,
    train_anomaly_model,
    detect_anomalies,
    predict_new_expense,
    visualize_anomalies,
    export_results
)


def main():
    print("=" * 55)
    print("       SMART EXPENSE ANOMALY DETECTION")
    print("=" * 55)

    try:
        df = load_expenses("expenses.csv")

        analyze_expenses(df)

        model = train_anomaly_model(df)

        results = detect_anomalies(model, df)

        predict_new_expense(
            model,
            amount=10000,
            hour=2,
            day=3,
            category="Shopping"
        )

        visualize_anomalies(results)
        export_results(results)

        print("\nProject completed successfully!")

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
