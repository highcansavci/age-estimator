import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple

def parse_image_filename(filename: str) -> Tuple[int, str, str, str]:
    """
    Parse image filename to extract metadata.
    Example filename format: [age]_[gender]_[race]_[date&time].jpg
    """
    # Remove .jpg extension and split by underscore
    name_parts = filename.replace('.jpg', '').split('_')
    
    age = int(name_parts[0])
    gender = name_parts[1]
    race = name_parts[2]
    
    return age, gender, race

def process_directory(directory_path: str) -> pd.DataFrame:
    """
    Process all jpg files in the directory and create a DataFrame.
    """
    data = []
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.jpg'):
            try:
                age, gender, race = parse_image_filename(filename)
                data.append({
                    'filename': filename,
                    'age': age,
                    'gender': gender,
                    'race': race
                })
            except (ValueError, IndexError) as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    return pd.DataFrame(data)

def create_train_test_split(df: pd.DataFrame, test_size: float = 0.2, 
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing sets.
    """
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
    )
    return train_df, test_df

def main():
    # Configure these parameters as needed
    input_directory = "../UTKface_inthewild"
    output_directory = "../dataset"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Process all images
    print("Processing images...")
    df = process_directory(input_directory)
    
    # Create train-test split
    print("Creating train-test split...")
    train_df, test_df = create_train_test_split(df)
    
    # Save to CSV files
    train_output_path = os.path.join(output_directory, "train.csv")
    test_output_path = os.path.join(output_directory, "test.csv")
    
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    
    print(f"Processing complete!")
    print(f"Total images processed: {len(df)}")
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")
    print(f"Files saved to: {output_directory}")

if __name__ == "__main__":
    main()