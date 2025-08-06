import sqlite3
import csv
from datetime import datetime

def csv_to_sqlite(csv_file, db_file):
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create the alumni table (adjust based on the structure of your CSV)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alumni (
        student_id INTEGER PRIMARY KEY,
        student_full_name TEXT,
        department_name_code TEXT,
        joining_year INTEGER,
        graduation_year INTEGER,
        contact_email TEXT,
        contact_phone_number TEXT,
        date_of_birth TEXT,
        student_address TEXT,
        student_city TEXT,
        academic_score_percentage REAL,
        attendance_percentage REAL,
        got_job_offer_in_campus_placement TEXT,
        job_offered_by_company TEXT,
        starting_campus_offer_value REAL,
        notes TEXT
    )
    ''')

    # Open the CSV file and read data
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        
        # Insert data into the alumni table
        for row in csvreader:
            student_id = int(row[0])
            student_full_name = row[1]
            department_name_code = row[2]
            joining_year = int(row[3])
            graduation_year = int(row[4])
            contact_email = row[5]
            contact_phone_number = row[6]
            date_of_birth = datetime.strptime(row[7], '%d-%m-%Y').date()
            student_address = row[8]
            student_city = row[9]
            academic_score_percentage = float(row[10])
            attendance_percentage = float(row[11])
            got_job_offer_in_campus_placement = row[12] == "Yes"
            job_offered_by_company = row[13] if row[13] else None
            starting_campus_offer_value = float(row[14]) if row[14] else None
            notes = row[15]
            
            cursor.execute('''
            INSERT INTO alumni (
                student_id, student_full_name, department_name_code, joining_year, 
                graduation_year, contact_email, contact_phone_number, date_of_birth, 
                student_address, student_city, academic_score_percentage, 
                attendance_percentage, got_job_offer_in_campus_placement, 
                job_offered_by_company, starting_campus_offer_value, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (student_id, student_full_name, department_name_code, joining_year, 
                  graduation_year, contact_email, contact_phone_number, date_of_birth, 
                  student_address, student_city, academic_score_percentage, 
                  attendance_percentage, got_job_offer_in_campus_placement, 
                  job_offered_by_company, starting_campus_offer_value, notes))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

# Example usage:
csv_to_sqlite('alumni_data.csv', 'AlumniDB.db')
