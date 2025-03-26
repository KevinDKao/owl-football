import pandas as pd
import os

CONFIG = "https://docs.google.com/spreadsheets/d/12dvkqj-NcjCYgoUs1CdCIj9OH4pEJyHZTe163_iEqds/edit?gid=161077302#gid=161077302"


def process_nfl_combine_data(
    input_file, output_file="nfl_combine_2025_combined.feather"
):
    """
    Parse NFL Combine Excel file and combine all relevant information into one Feather file.

    Args:
        input_file (str): Path to the NFL Combine Excel file
        output_file (str): Path to save the combined Feather file
    """
    try:
        # Read the Excel file
        xl = pd.ExcelFile(input_file)

        # Get all sheet names
        sheet_names = xl.sheet_names

        # Initialize an empty DataFrame to store combined data
        combined_data = pd.DataFrame()

        # Process each sheet
        for sheet_name in sheet_names:
            # Read the sheet
            df = pd.read_excel(input_file, sheet_name=sheet_name)

            # Skip empty sheets
            if df.empty:
                continue

            # Add a column to identify the source sheet. The pages were originally organized by position.
            df["position"] = sheet_name

            # Append to the combined data
            if combined_data.empty:
                combined_data = df
            else:
                # Handle different column structures
                combined_data = pd.concat(
                    [combined_data, df], ignore_index=True, sort=False
                )

        # Clean and standardize the combined data
        # Remove any completely empty rows or columns
        combined_data = combined_data.dropna(how="all").dropna(how="all", axis=1)

        # Save to Feather
        combined_data.to_csv(output_file)

        return True

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False


# Example usage
if __name__ == "__main__":
    process_nfl_combine_data("./raw.xlsx")
