import random
import numpy as np
import pandas as pd
from datetime import datetime , timedelta
# generate random phone numbers
def generate_phone_number(used_phone_numbers):
    while True:
        phone_number = "000000{}{}{}{}{}{}".format(random.randint(0,9), random.randint(0,9), random.randint(0,9), random.randint(0,9), random.randint(0,9), random.randint(0,9))
        if phone_number not in used_phone_numbers:
            used_phone_numbers.add(phone_number)
            return np.int64(phone_number)

# Generate first names
def generate_unique_names(used_names, num_names):
    unique_names = set()
    while len(unique_names) < num_names:
        first_name = random.choice(first_name_pool)
        last_name = random.choice(last_name_pool)
        full_name = (first_name, last_name)
        if full_name not in used_names:
            used_names.add(full_name)
            unique_names.add(full_name)
    return unique_names
# Generate time
def generate_time_slots():
    start_time = datetime.strptime("07:00", "%H:%M")
    end_time = datetime.strptime("21:45", "%H:%M")
    time_slots = []
    while start_time <= end_time:
        time_slots.append(start_time.strftime('%H:%M'))
        start_time += timedelta(minutes=15)
    return time_slots
for lx in range(10,32):
    first_name_pool = ["Liam", "Olivia", "Ava", "Ethan", "Sophia", "Ella", "Emma", "James", "Oliver",
        "Sophie", "Benjamin", "Isabella", "Mia", "William", "Avery", "Natalie", "Logan", "Jacob",
        "Evelyn", "Daniel", "Sophie", "Sophia", "Oliver", "Elijah", "Aiden", "Lucy", "Matthew", "Grace", 
        "Chloe", "Jackson", "Elena", "Hannah", "Carter", "Victoria", "Connor", "Luna", "Gabriel", "Zoe", 
        "Nathan", "Penelope", "Levi", "Audrey", "Dylan", "Madeline", "Ryan", "Harper", 
        "Michael", "Aria", "Eli", "Stella", "Henry", "Mila", "David", "Maya", "Luke", "Eva", "Christopher", 
        "Amelia", "Julian", "Clara","Alexander", "Mason", "Eleanor", "Aurora", "Asher", "Liam", "Penelope", "William","Mateo", "Elena", 
        "Zoey", "Luna", "Ezra", "Mila", "Layla", "Jack", "Harper", "Julian", "Madison", "Amelia",
        "Sebastian", "Scarlett", "Lincoln", "Evelyn", "Lucas", "Camila", "Eva", "Avery", "Isabella",
        "Liliana", "Samuel", "Grace", "Caleb", "Ariana", "Eliana", "Adam", "Xavier", "Kaylee", "Isabelle",
        "Gabriella", "Nora", "Riley", "Leah", "Sofia", "Hazel", "Stella", "Mason", "Nova", "Natalie",
        "Jameson", "Audrey", "Anthony", "Joseph", "Piper", "Elizabeth", "Evelyn", "Genesis", "Adrian",
        "Eleanor", "Brooklyn", "Nolan", "Nicholas", "Caroline", "Cora", "Bella", "Hudson", "Katherine",
        "Zachary", "Violet", "Claire", "Ellie", "Landon", "Lillian", "Charlie", "Nathan", "Rebecca",
        "Naomi", "Emilia", "Jasmine", "Zara", "Ezekiel", "Delilah", "Jaxon", "Sarah", "Emma"]
    last_name_pool = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller",
        "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Clark", "Lewis", "Lee", 
        "Walker", "Hall", "Allen", "Young", "Martinez", "Robinson", "Garcia", "Martinez", "Lopez", "Hill", "Scott", 
        "Green", "Adams", "Baker", "Gonzalez", "Nelson", "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips", 
        "Campbell", "Parker", "Evans", "Edwards", "Collins", "Stewart", "Sanchez", "Morris", "Rogers", "Reed", "Cook", 
        "Morgan", "Cooper", "Rivera", "Reynolds", "King", "Foster", "Fisher", "Wells", "Barnes","Kim", "Yang", "Park", 
        "Jung", "Choi", "Chung", "Yoon", "Kang", "Shin", "Song", 
        "Oh", "Jin", "Moon", "Kwon", "Seo", "Yoo", "Hwang", "Lim", "Lee", "Jeong",
        "Han", "Lee", "Kim", "Park", "Jung", "Choi", "Chung", "Yoon", "Kang", "Shin",
        "Song", "Oh", "Jin", "Moon", "Kwon", "Seo", "Yoo", "Hwang", "Lim", "Lee",
        "Jeong", "Han", "Lee", "Cho", "Yi", "Ryu", "Yoon", "Baek", "Ha", "Nam",
        "Cha", "Do", "Chang", "Suh", "Sun", "Sung", "Woo", "Kim", "Hahn", "Shin"]

    # Generate domains for emails
    domains = ["null.com"]

    # Generate a list of random "FU" and "NP" entries
    types = ["FU", "NP"]

    # Create sets to store used names, emails, and phone numbers
    used_names = set()
    used_emails = set()
    used_phone_numbers = set()

    # Generate unique names
    unique_names = generate_unique_names(used_names, 60)

    # Generate emails and phone numbers
    email_name_map = {}  # Dictionary to map each email to its respective first and last name
    for first_name, last_name in unique_names:
        # Generate email
        while True:
            separator = random.choice([".", "_"])
            email = "{}{}{}@{}".format(first_name.lower(), separator, last_name.lower(), random.choice(domains))
            if email not in used_emails:
                used_emails.add(email)
                email_name_map[email] = (first_name, last_name)
                break

        # Generate phone number
        phone_number = generate_phone_number(used_phone_numbers)
        birthday = datetime(1982, 12, 28)
        # Randomly select type
        t = random.choice(types)
        doctor_name = "Ehab Yacoub"



        # Output the data



    time_slots = generate_time_slots()
    # Create a DataFrame
    data = pd.DataFrame({
        "first name": [email_name_map[email][0] for email in used_emails],
        "last name": [email_name_map[email][1] for email in used_emails],
        "Date of birth": [np.datetime64("1982-12-28") for _ in range(60)],
        "email": list(used_emails),
        "phone number": list(used_phone_numbers),
        "": "",
        "doctor name": [doctor_name for _ in range(60)],
        "type": [random.choice(types) for _ in range(60)],
        "date": [np.datetime64(f"2024-05-{lx}") for _ in range(60)],
        "time": time_slots
    })
    data['date'] = pd.to_datetime(data['date'])


    # Save DataFrame to Excel


    with pd.ExcelWriter(f"C:\\Users\\kerol\\OneDrive\\Desktop\\test\\Ehab 05-{lx}-2024.xlsx", engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook  = writer.book
        worksheet = writer.sheets['Sheet1']

        # Loop through the column and apply date format to each cell
        date_format = workbook.add_format({'num_format': 'mm-dd-yyyy','align': 'left'})
        for idx, value in enumerate(data['date'], start=1):
            worksheet.write(idx, 8, value, date_format)
            # Loop through the column and apply date format to each cell
        date_birth_format = workbook.add_format({'num_format': 'm-d-yyyy','align': 'right'})
        for idx, value in enumerate(data['Date of birth'], start=1):
            worksheet.write(idx, 2, value, date_birth_format)

        time_format = workbook.add_format({'num_format': 'h:mm','align': 'left'})
        for idx, value in enumerate(data['time'], start=1):
            worksheet.write(idx, 9, value, time_format)
