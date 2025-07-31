# FICO Score Quantization

This project maps FICO credit scores to credit risk ratings using quantization techniques. It provides a scalable method for discretizing continuous credit scores into interpretable categories for modeling or analysis.

## Files

- `fico_quantization_script.py`: Main Python script that performs the quantization.
- `fico_rating_map.csv`: Maps each borrower's FICO score to a quantized rating.
- `bucket_summary.csv`: Summary statistics for each rating bucket, including default rates.

## How It Works

1. **Input**: `Task 3 and 4_Loan_Data.csv` — Contains borrower info, including FICO scores and default labels.
2. **Quantization**: Uses quantiles to divide FICO scores into 10 rating buckets (0 = best, 9 = worst).
3. **Output**:
   - `fico_rating_map.csv`: Contains columns:
     - `fico_score`: Original credit score
     - `quantile_rating`: Assigned rating (0 to 9)
     - `default`: Whether the borrower defaulted
   - `bucket_summary.csv`: Contains:
     - `quantile_rating`: Rating level
     - `min_fico` / `max_fico`: FICO score range
     - `avg_fico`: Average score in the bucket
     - `default_rate`: % of defaults in the bucket
     - `count`: Number of records

## Usage

To run the script:

```bash
python fico_quantization_script.py
```

Make sure `Task 3 and 4_Loan_Data.csv` is in the same directory or update the path in the script.

## Example Output (Bucket Summary)

| Rating | FICO Range | Avg FICO | Default Rate |
|--------|------------|----------|---------------|
| 0      | 715–850    | ~742     | ~3.6%         |
| 9      | 408–560    | ~530     | ~49.1%        |

## License

MIT License
