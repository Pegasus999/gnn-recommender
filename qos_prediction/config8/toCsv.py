#load a txt file and turn it into a csv file
import numpy as np

def to_csv(input_file: str, output_file: str):
      """
      Load a text file with numerical data and save it as a CSV file.
      
      Parameters:
      input_file (str): Path to the input text file.
      output_file (str): Path to the output CSV file.
      """
      try:
         # Load the data from the text file
         data = np.loadtxt(input_file)
         
         # Save the data to a CSV file
         np.savetxt(output_file, data, delimiter=',', fmt='%.6f')
         print(f"Data successfully saved to {output_file}")
         
      except Exception as e:
         print(f"Error processing files: {e}")

# Example usage
if __name__ == "__main__":
    input_file = "rtMatrix.txt"  # Replace with your input file
    output_file = "rtMatrix.csv"  # Desired output CSV file
    to_csv(input_file, output_file)