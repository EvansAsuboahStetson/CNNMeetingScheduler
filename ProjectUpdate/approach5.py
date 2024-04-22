from datetime import date as dt_date

from keras.src.saving.saving_api import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

from ProjectUpdate.Persona import Persona

from datetime import date, timedelta

from datetime import timedelta
import numpy as np

from sklearn.model_selection import train_test_split

from ProjectUpdate.newCNNModel import MeetingSchedulerModel

np.set_printoptions(threshold=np.inf)

from datetime import datetime
import datetime


def generate_random_time_slot():
    start_hour = random.randint(10, 15)
    start_minute = random.choice([0, 15, 30, 45])
    start_time = f"{start_hour:02d}:{start_minute:02d}"

    # Assuming a 1-hour meeting duration
    end_hour = start_hour + random.randint(1, 2)
    end_time = f"{end_hour:02d}:{start_minute:02d}"
    return start_time, end_time


def generate_engineer_meetings(start_year, end_year):
    meetings = []
    current_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)

    while current_date <= end_date:
        if current_date.weekday() < 5:  # Stand-Up Meetings on Weekdays
            meetings.append({"start_date": current_date, "frequency": "once",
                             "event": "Daily Stand-Up", "start_time": "09:00", "end_time": "09:30"})

        if current_date.weekday() == 1 and (
                current_date - date(current_date.year, current_date.month, 1)).days % 14 == 0:  # Biweekly on Tuesdays
            meetings.append({"start_date": current_date, "frequency": "once",
                             "event": "Project Review", "start_time": "10:00", "end_time": "11:00"})

        if current_date.weekday() == 0 and current_date.day <= 7:  # First Monday of the Month
            meetings.append({"start_date": current_date, "frequency": "once",
                             "event": "Client Meeting", "start_time": "14:00", "end_time": "15:00"})

        if current_date.day == 1 and current_date.month % 3 == 0:  # First Day of Every Third Month
            meetings.append({"start_date": current_date, "frequency": "once",
                             "event": "Quarterly Strategy Session", "start_time": "10:45", "end_time": "12:00"})

        if random.random() < 0.10:  # Roughly 5% chance of a meeting on any given day
            start_time, end_time = generate_random_time_slot()
            meetings.append({"start_date": current_date, "frequency": "once",
                             "event": "Meeting with Colleagues", "start_time": start_time, "end_time": end_time})

        current_date += timedelta(days=1)

    return meetings


import random


def generate_random_time_slot_within_availability(availability, meeting_duration_hours=1):
    start_availability_hour, start_availability_minute = availability[0]
    end_availability_hour, end_availability_minute = availability[1]

    # Convert availability to minutes
    start_availability_total_minutes = start_availability_hour * 60 + start_availability_minute
    end_availability_total_minutes = end_availability_hour * 60 + end_availability_minute

    # Convert meeting duration to minutes
    meeting_duration_minutes = int(meeting_duration_hours * 60)

    # Adjust end time to accommodate meeting duration
    adjusted_end_time_minutes = end_availability_total_minutes - meeting_duration_minutes

    # Check if the meeting can be accommodated
    if start_availability_total_minutes >= adjusted_end_time_minutes:
        raise ValueError("Invalid availability window or meeting duration")

    # Generate a random start time within the adjusted time window
    random_start_minutes = random.choice(range(start_availability_total_minutes, adjusted_end_time_minutes + 1, 15))
    end_time_minutes = random_start_minutes + meeting_duration_minutes

    # Convert start and end times back to hours and minutes
    start_hour, start_minute = divmod(random_start_minutes, 60)
    end_hour, end_minute = divmod(end_time_minutes, 60)

    start_time = f"{start_hour:02d}:{start_minute:02d}"
    end_time = f"{end_hour:02d}:{end_minute:02d}"

    return start_time, end_time


def generate_meetings_for_persona(start_year, end_year, persona, meeting_types):
    meetings = []
    current_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)

    while current_date <= end_date:
        for meeting_type in meeting_types:
            if meeting_type['condition'](current_date, persona):
                if 'start_time' in meeting_type and 'end_time' in meeting_type:
                    start_time = meeting_type['start_time']
                    end_time = meeting_type['end_time']
                else:
                    duration_hours = meeting_type.get('duration', 1)
                    start_time, end_time = generate_random_time_slot_within_availability(persona.availability,
                                                                                         duration_hours)

                meetings.append({
                    "start_date": current_date,
                    "frequency": "once",
                    "event": meeting_type['event'],
                    "start_time": start_time,
                    "end_time": end_time
                })
        if random.random() < 0.10:
            start_time, end_time = generate_random_time_slot_within_availability(persona.availability)
            meetings.append({
                "start_date": current_date,
                "frequency": "once",
                "event": "Random Meeting",
                "start_time": start_time,
                "end_time": end_time
            })

        current_date += timedelta(days=1)

    return meetings


def create_schedule_matrix(persona_calendar):
    schedule_matrix = {}
    for year in persona_calendar.keys():
        for month_str in persona_calendar[year].keys():
            for date_str, day_schedule in persona_calendar[year][month_str].items():
                daily_schedule = [1 if day_schedule[time_slot] else 0 for time_slot in day_schedule]
                schedule_matrix[date_str] = daily_schedule
    return schedule_matrix


def encode_day_of_week(day):
    # Returns a one-hot encoded vector for the day of the week
    return np.eye(7)[day]


def encode_time_of_day():
    # Returns a cyclic encoding for each time slot of the day
    times_per_day = 96  # Assuming there are 96 time slots per day (15 minutes each)
    return np.array(
        [(np.sin(2 * np.pi * i / times_per_day), np.cos(2 * np.pi * i / times_per_day)) for i in range(times_per_day)])


def create_period_schedule_matrix(persona_schedule, start_date, num_days=28):
    timeslots_per_day = 96  # Assuming 15-minute intervals throughout a 24-hour period
    features_per_slot = 10  # Including time of day (2), availability (1), day of week (7)
    matrix = np.zeros((num_days, timeslots_per_day, features_per_slot))

    time_of_day_encoded = encode_time_of_day()  # Cyclic encoding for time of day

    for day_idx in range(num_days):
        current_date = start_date + timedelta(days=day_idx)
        day_of_week_encoded = encode_day_of_week(current_date.weekday())
        day_schedule = persona_schedule.get(current_date.isoformat(), {})

        for time_idx in range(timeslots_per_day):
            time_slot_key = f"{time_idx // 4:02d}:{(time_idx % 4) * 15:02d}"  # Format time slot key e.g., '09:30'
            # Time of day encoding
            matrix[day_idx, time_idx, :2] = time_of_day_encoded[time_idx]
            # Check specific time slot for event
            event = day_schedule.get(time_slot_key)
            if event is not None:
                matrix[day_idx, time_idx, 2] = 1

            else:
                matrix[day_idx, time_idx, 2] = 0
            # Day of the week encoding
            matrix[day_idx, time_idx, 3:] = day_of_week_encoded

    return matrix


doctor = Persona(
    name="Doctor",
    availability=((9, 0), (17, 0)),  # Available from 9:00 to 17:00
    preferred_meeting_times=[((10, 0), (12, 0)), ((15, 0), (17, 0))],
    # Prefers meetings between 10:00-12:00 and 15:00-17:00
    busy_periods=[((12, 0), (13, 0))],  # Busy during 12:00-13:00 (lunch break)
    preferred_days=["Monday", "Wednesday", "Thursday", "Friday"],
    busy_days=["Tuesday"],
    start_year=2023,
    end_year=2035
)

# Define the teacher persona
teacher = Persona(
    name="Teacher",
    availability=((8, 0), (16, 0)),
    preferred_meeting_times=[((15, 0), (16, 0))],
    busy_periods=[((12, 0), (13, 0))],  # Busy during 12:00-13:00 (lunch break)
    preferred_days=["Monday", "Wednesday", "Thursday", "Friday"],
    busy_days=["Tuesday", "Saturday", "Sunday"],  # Busy on weekends
    start_year=2023,
    end_year=2035
)

# Define the engineer persona
engineer = Persona(
    name="Engineer",
    availability=((9, 0), (18, 0)),  # Available from 9:00 to 18:00
    preferred_meeting_times=[((10, 0), (12, 0)), ((14, 0), (16, 0))],
    busy_periods=[],  # No specific busy periods
    preferred_days=["Tuesday", "Thursday", "Friday", "Saturday"],
    busy_days=["Monday"],  # Busy on weekends
    start_year=2023,
    end_year=2035
)

lawyer = Persona(
    name="Lawyer",
    availability=((8, 0), (18, 0)),  # Available from 8:00 to 18:00
    preferred_meeting_times=[((9, 0), (11, 0)), ((14, 0), (16, 0))],
    # Prefers meetings in the late morning and mid-afternoon
    busy_periods=[((12, 0), (13, 0))],  # Lunch break at 12:00-13:00
    preferred_days=["Monday", "Wednesday", "Thursday", "Friday"],
    busy_days=["Tuesday", "Thursday"],  # Weekends off
    start_year=2023,
    end_year=2035
)

developer = Persona(
    name="Software Developer",
    availability=((9, 0), (17, 0)),  # Typical office hours
    preferred_meeting_times=[((10, 0), (11, 0)), ((15, 0), (16, 0))],  # Mid-morning and late afternoon
    busy_periods=[],  # Flexible breaks
    preferred_days=["Monday", "Wednesday", "Friday", "Saturday"],
    busy_days=["Tuesday", "Thursday"],
    start_year=2023,
    end_year=2035
)

consultant = Persona(
    name="Consultant",
    availability=((10, 0), (18, 0)),  # Available from 10:00 to 18:00
    preferred_meeting_times=[((11, 0), (13, 0)), ((15, 0), (17, 0))],  # Morning and late afternoon
    busy_periods=[((13, 0), (14, 0))],  # Lunch break at 13:00-14:00
    preferred_days=["Monday", "Wednesday", "Friday"],
    busy_days=["Saturday", "Sunday", "Thursday"],
    start_year=2023,
    end_year=2035
)
designer = Persona(
    name="Designer",
    availability=((9, 30), (17, 30)),  # Available from 9:30 to 17:30
    preferred_meeting_times=[((10, 0), (12, 0)), ((14, 0), (16, 0))],  # Morning and early afternoon
    busy_periods=[],  # No specific busy periods
    preferred_days=["Monday", "Wednesday", "Friday"],
    busy_days=["Tuesday", "Thursday", "Saturday", "Sunday"],
    start_year=2023,
    end_year=2035
)

marketing_specialist = Persona(
    name="Marketing Specialist",
    availability=((8, 30), (17, 30)),  # Available from 8:30 to 17:30
    preferred_meeting_times=[((9, 0), (11, 0)), ((14, 0), (16, 0))],  # Late morning and mid-afternoon
    busy_periods=[((8, 0), (9, 0))],  # Lunch break at 12:00-13:00
    preferred_days=["Monday", "Wednesday", "Thursday", "Friday"],
    busy_days=["Saturday", "Sunday"],
    start_year=2023,
    end_year=2035
)
fitness_trainer = Persona(
    name="Fitness Trainer",
    availability=((6, 0), (20, 0)),  # Early mornings to evenings
    preferred_meeting_times=[((6, 0), (9, 0)), ((17, 0), (20, 0))],  # Prefer early mornings and evenings for sessions
    busy_periods=[((13, 0), (15, 0))],  # Midday break
    preferred_days=["Monday", "Tuesday", "Thursday", "Friday"],
    busy_days=["Saturday", "Sunday", "Wednesday"],
    start_year=2023,
    end_year=2035
)

freelancer = Persona(
    name="Freelancer",
    availability=((8, 0), (22, 0)),  # Flexible working hours
    preferred_meeting_times=[((11, 0), (13, 0))],  # Prefers late morning meetings
    busy_periods=[],  # Varies based on project
    preferred_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    busy_days=[],  # Can work any day depending on project
    start_year=2023,
    end_year=2035
)
construction_manager = Persona(
    name="Construction Project Manager",
    availability=((6, 0), (15, 0)),  # Early start to avoid late hours
    preferred_meeting_times=[((7, 0), (9, 0))],  # Early meetings before site work begins
    busy_periods=[((14, 0), (16, 0))],  # Short lunch break
    preferred_days=["Monday", "Wednesday", "Friday"],
    busy_days=["Saturday"],  # Often on site on weekends
    start_year=2023,
    end_year=2035
)
research_scientist = Persona(
    name="Research Scientist",
    availability=((8, 0), (17, 0)),  # Typical academic hours
    preferred_meeting_times=[((10, 0), (12, 0)), ((15, 0), (16, 0))],  # Mid-morning and mid-afternoon
    busy_periods=[],  # Flexibility depending on experiments
    preferred_days=["Monday", "Tuesday", "Thursday"],
    busy_days=[],  # Works any weekday depending on research needs
    start_year=2023,
    end_year=2035
)

retail_manager = Persona(
    name="Retail Store Manager",
    availability=((10, 0), (18, 0)),  # Standard retail hours
    preferred_meeting_times=[((11, 0), (12, 0))],  # Late morning before the rush
    busy_periods=[((13, 0), (14, 0))],  # Lunchtime rush
    preferred_days=["Tuesday", "Wednesday", "Thursday"],
    busy_days=["Saturday", "Sunday"],  # Peak shopping days
    start_year=2023,
    end_year=2035
)

startup_ceo = Persona(
    name="Startup CEO",
    availability=((8, 0), (20, 0)),  # Long hours
    preferred_meeting_times=[((9, 0), (10, 0)), ((17, 0), (19, 0))],  # Early meetings or late evening wrap-ups
    busy_periods=[((12, 0), (13, 0))],  # Quick lunch breaks
    preferred_days=["Monday", "Wednesday", "Friday"],
    busy_days=["Saturday"],  # Often busy with networking on weekends
    start_year=2023,
    end_year=2035
)

event_planner = Persona(
    name="Event Planner",
    availability=((10, 0), (19, 0)),  # Later start, works evenings
    preferred_meeting_times=[((11, 0), (13, 0)), ((16, 0), (18, 0))],  # Midday and late afternoon
    busy_periods=[],  # Very flexible but often out on site visits
    preferred_days=["Tuesday", "Thursday", "Saturday"],
    busy_days=["Sunday"],  # Major event days often fall on weekends
    start_year=2023,
    end_year=2035
)

hr_manager = Persona(
    name="HR Manager",
    availability=((9, 0), (17, 0)),  # Office hours
    preferred_meeting_times=[((10, 0), (11, 0)), ((15, 0), (16, 0))],  # Late morning and late afternoon
    busy_periods=[((12, 0), (13, 0))],  # Lunch break
    preferred_days=["Monday", "Tuesday", "Thursday"],
    busy_days=["Wednesday", "Friday"],  # Busy with workshops and training sessions
    start_year=2023,
    end_year=2035
)

doctor_meetings = generate_meetings_for_persona(2024, 2034, doctor, [
    {
        "event": "Patient Appointments",
        "condition": lambda current_date, persona: current_date.weekday() in [0, 2, 3, 4],
        "start_time": "10:00",
        "end_time": "12:00"
    },
    {
        "event": "Staff Meetings",
        "condition": lambda current_date, persona: current_date.weekday() == 2 and (
                current_date - dt_date(current_date.year, current_date.month, 1)).days % 14 == 0,
        "start_time": "15:00",
        "end_time": "16:30"
    },
    {
        "event": "Medical Seminar",
        "condition": lambda current_date, persona: current_date.weekday() == 3 and current_date.day <= 7,
        "start_time": "15:00",
        "end_time": "17:00"
    },
    {
        "event": "Annual Medical Conference",
        "condition": lambda current_date, persona: current_date.month == 4 and current_date.day == 1,
        "start_time": "09:00",
        "end_time": "12:00"
    },
    {
        "event": "Morning Consultations",
        "condition": lambda current_date, persona: current_date.weekday() in [0, 2, 3, 4],
        "start_time": "09:00",
        "end_time": "10:00"
    },
    {
        "event": "Lunch Time Consultations",
        "condition": lambda current_date, persona: current_date.weekday() < 5 and current_date.weekday() != 1,
        "start_time": "13:00",
        "end_time": "14:00"
    },
    {
        "event": "Afternoon Consultations",
        "condition": lambda current_date, persona: current_date.weekday() in [0, 2, 3, 4],
        "start_time": "15:00",
        "end_time": "17:00"
    },
    {
        "event": "Special Case Reviews",
        "condition": lambda current_date, persona: current_date.weekday() == 4,
        "start_time": "15:00",
        "end_time": "17:00"
    },
    {
        "event": "Weekend Emergency Hours",
        "condition": lambda current_date, persona: current_date.weekday() > 4,
        "start_time": "10:00",
        "end_time": "12:00"
    }
])
teacher_meetings = generate_meetings_for_persona(2024, 2034, teacher, [
    {
        "event": "Class Sessions",
        "condition": lambda current_date, persona: current_date.weekday() in [0, 2, 3, 4],
        "start_time": "08:00",
        "end_time": "12:00"
    },
    {
        "event": "Parent-Teacher Meetings",
        "condition": lambda current_date, persona: current_date.weekday() == 3 and 15 <= current_date.day <= 21,
        "start_time": "15:00",
        "end_time": "16:00"
    },
    {
        "event": "Professional Development",
        "condition": lambda current_date, persona: current_date.month % 4 == 0 and current_date.day == 1,
        "start_time": "09:00",
        "end_time": "12:00"
    },
    {
        "event": "Morning Classes",
        "condition": lambda current_date, persona: current_date.weekday() in [0, 2, 3, 4],
        "start_time": "08:00",
        "end_time": "12:00"
    },
    {
        "event": "Afternoon Classes",
        "condition": lambda current_date, persona: current_date.weekday() in [0, 2, 3, 4],
        "start_time": "13:00",
        "end_time": "15:00"
    },
    {
        "event": "Extra Tutoring Sessions",
        # This event must be adjusted to fit within working hours.
        "condition": lambda current_date, persona: current_date.weekday() == 4,
        "start_time": "15:00",
        "end_time": "16:00"
    },
    {
        "event": "Weekend Grading Hours",
        "condition": lambda current_date, persona: current_date.weekday() > 4,
        "start_time": "10:00",
        "end_time": "12:00"
    }
])

engineer_meetings = generate_meetings_for_persona(2024, 2034, engineer, [
    {
        "event": "Daily Stand-Ups",
        "condition": lambda current_date, persona: current_date.weekday() in [1, 3, 4, 5],  # Weekdays except Monday
        "start_time": "09:00",
        "end_time": "09:15"
    },
    {
        "event": "Project Reviews",
        "condition": lambda current_date, persona: current_date.weekday() == 2 and (
                current_date - dt_date(current_date.year, current_date.month, 1)).days % 14 == 0,
        # Adjusted to ensure it only occurs on preferred days
        "start_time": "10:00",
        "end_time": "11:00"
    },
    {
        "event": "Client Meetings",
        "condition": lambda current_date, persona: current_date.weekday() == 4 and current_date.day <= 7,
        # First Friday of the month; within preferred meeting times
        "start_time": "14:00",
        "end_time": "16:00"
    },
    {
        "event": "Team-Building Events",
        "condition": lambda current_date, persona: (current_date.weekday() in [1, 3, 4, 5]) and (
                current_date.month % 6 == 0) and (current_date.day == 15),
        "start_time": "10:00",
        "end_time": "14:00"
    },
    {
        "event": "Morning Stand-Up",
        "condition": lambda current_date, persona: current_date.weekday() in [1, 3, 4, 5],
        "start_time": "09:00",
        "end_time": "09:30"
    },
    {
        "event": "Weekly Project Sync",
        "condition": lambda current_date, persona: current_date.weekday() == 2,
        # Ensures this only happens on Wednesdays which are preferred days
        "start_time": "10:00",
        "end_time": "11:00"
    },
    {
        "event": "Bi-Weekly Client Update Call",
        "condition": lambda current_date, persona: current_date.weekday() == 4 and (
                current_date - dt_date(current_date.year, current_date.month, 1)).days % 14 == 0,
        # Adjusted to every other preferred Friday
        "start_time": "14:00",
        "end_time": "15:00"
    },
    {
        "event": "End-of-Day Wrap-Up",
        "condition": lambda current_date, persona: current_date.weekday() in [1, 3, 4, 5],
        "start_time": "17:00",
        "end_time": "17:30"
    }
])

lawyer_meetings = generate_meetings_for_persona(2024, 2034, lawyer, [
    {
        "event": "Client Consultations",
        "condition": lambda current_date, persona: current_date.weekday() in [0, 2, 4],  # Monday, Wednesday, Friday
        "start_time": "10:00",
        "end_time": "12:00"
    },
    {
        "event": "Legal Research",
        "condition": lambda current_date, persona: current_date.weekday() == 4,  # Fridays
        "start_time": "09:00",
        "end_time": "11:00"
    },
    {
        "event": "Legal Briefings",
        "condition": lambda current_date, persona: current_date.weekday() == 2,  # Wednesdays
        "start_time": "14:00",
        "end_time": "16:00"
    },
    {
        "event": "Strategy Meetings",
        "condition": lambda current_date, persona: current_date.weekday() == 0,  # Mondays
        "start_time": "14:00",
        "end_time": "16:00"
    },
    {
        "event": "Case Review Sessions",
        "condition": lambda current_date, persona: current_date.weekday() == 0,  # Mondays
        "start_time": "09:00",
        "end_time": "11:00"
    },
    {
        "event": "Witness Preparation",
        "condition": lambda current_date, persona: current_date.weekday() == 2,  # Wednesdays
        "start_time": "09:00",
        "end_time": "11:00"
    }
])

developer_meetings = generate_meetings_for_persona(2024, 2034, developer, [
    {
        "event": "Scrum Meeting",
        "condition": lambda current_date, persona: current_date.weekday() < 5,  # Daily on weekdays
        "start_time": "09:30",
        "end_time": "10:00"
    },
    {
        "event": "Code Review",
        "condition": lambda current_date, persona: current_date.weekday() == 2,  # Every Wednesday
        "start_time": "15:00",
        "end_time": "16:00"
    },
    {
        "event": "Client Demo",
        "condition": lambda current_date, persona: current_date.weekday() == 4,  # Every Friday
        "start_time": "14:00",
        "end_time": "15:00"
    }

])

consultant_meetings = generate_meetings_for_persona(2024, 2034, consultant, [
    {
        "event": "Client Consultations",
        "condition": lambda current_date, persona: current_date.weekday() in [0, 2, 4],  # Monday, Wednesday, Friday
        "start_time": "11:00",
        "end_time": "13:00"  # Already fits within preferred times
    },
    {
        "event": "Project Presentations",
        "condition": lambda current_date, persona: current_date.weekday() < 3,  # Adjusted to Monday to Wednesday
        "start_time": "15:00",
        "end_time": "17:00"  # Already fits within preferred times
    },
    {
        "event": "Networking Events",
        # Check if these can be held on preferred days within the late afternoon slot.
        "condition": lambda current_date, persona: current_date.month in [6, 9] and current_date.weekday() in [0, 2, 4],
        "start_time": "15:00",
        "end_time": "17:00"  # Adjusted to end within the available time
    },
    {
        "event": "Business Strategy Discussion",
        # Adjusted as it overlaps with the lunch break.
        "condition": lambda current_date, persona: current_date.weekday() == 2,  # Every Wednesday
        "start_time": "15:00",
        "end_time": "17:00"  # Moved to a later time slot to avoid lunch break
    }
])

designer_meetings = generate_meetings_for_persona(2024, 2034, designer, [
    {
        "event": "Client Design Reviews",
        "condition": lambda current_date, persona: current_date.weekday() in [0, 2, 4],  # Already correct
        "start_time": "10:00",
        "end_time": "12:00"
    },

    {
        "event": "Portfolio Updates",
        # Ensure it happens on a preferred day within preferred times.
        "condition": lambda current_date, persona: current_date.month in [3, 9] and current_date.weekday() in [0, 2, 4],
        "start_time": "15:00",
        "end_time": "16:00"  # Adjusted to fit within preferred times
    },
    {
        "event": "Creative Workshops",
        "condition": lambda current_date, persona: current_date.weekday() == 2,  # Wednesdays
        "start_time": "14:00",
        "end_time": "16:00"  # Within preferred times
    },
    {
        "event": "Team Meetings",
        "condition": lambda current_date, persona: current_date.weekday() == 4,  # Fridays
        "start_time": "14:00",
        "end_time": "16:00"  # Within preferred times
    },
    {
        "event": "Inspirational Seminars",
        "condition": lambda current_date, persona: current_date.weekday() == 0,  # Mondays
        "start_time": "14:00",
        "end_time": "16:00"  # Within preferred times
    }
])

marketing_specialist_meetings = generate_meetings_for_persona(2024, 2034, marketing_specialist, [
    {
        "event": "Marketing Campaign Planning",
        "condition": lambda current_date, persona: current_date.weekday() == 0,  # Mondays
        "start_time": "09:00",
        "end_time": "11:00"  # Fits within preferred time
    },
    {
        "event": "Social Media Strategy Meetings",
        "condition": lambda current_date, persona: current_date.weekday() == 2,  # Wednesdays
        "start_time": "14:00",
        "end_time": "16:00"  # Fits within preferred time
    },
    {
        "event": "Market Research Analysis",
        "condition": lambda current_date, persona: current_date.weekday() == 4,  # Fridays
        "start_time": "11:00",
        "end_time": "13:00"  # Needs adjustment to avoid lunch break
    },
    {
        "event": "Content Brainstorming Session",
        "condition": lambda current_date, persona: current_date.weekday() == 0,  # Mondays
        "start_time": "11:00",
        "end_time": "12:00"  # Adjusted to end before lunch
    },
    {
        "event": "Client Pitch Presentation",
        "condition": lambda current_date, persona: current_date.weekday() == 2,  # Wednesdays
        "start_time": "16:00",
        "end_time": "17:00"  # Adjusted to fit within available hours
    },
    {
        "event": "Weekly Marketing Review",
        "condition": lambda current_date, persona: current_date.weekday() == 4,  # Fridays
        "start_time": "14:00",
        "end_time": "15:00"  # Adjusted to fit preferred time
    },
    {
        "event": "Team Building Activity",
        "condition": lambda current_date, persona: current_date.month % 3 == 0 and current_date.weekday() == 4,
        # Every third month on Friday
        "start_time": "15:00",
        "end_time": "16:00"  # Adjusted to fit within preferred time
    }
])

fitness_trainer_meetings = generate_meetings_for_persona(2024, 2034, fitness_trainer, [
    {
        "event": "Personal Training Sessions",
        "condition": lambda current_date, persona: current_date.weekday() in [0, 1, 3, 4],
        # Adjusted to preferred days only
        "start_time": "06:00",
        "end_time": "09:00"  # Within preferred meeting times
    },
    {
        "event": "Group Fitness Classes",
        "condition": lambda current_date, persona: current_date.weekday() in [1, 3],  # Tuesdays and Thursdays
        "start_time": "17:00",
        "end_time": "20:00"  # Within preferred meeting times
    },
    {
        "event": "Workout Planning",
        "condition": lambda current_date, persona: current_date.weekday() == 4,  # Fridays
        "start_time": "06:00",
        "end_time": "08:00"  # Adjusted to fit within the early morning preferred times
    },
    {
        "event": "Health Workshop",
        "condition": lambda current_date, persona: (
                                                           current_date.month % 3 == 0 and current_date.day <= 7) and current_date.weekday() in [
                                                       0, 1, 3, 4],
        # First week of every third month, adjusted to occur only on preferred days
        "start_time": "17:00",
        "end_time": "20:00"  # Fitting within the evening preferred times
    }
])

freelancer_meetings = generate_meetings_for_persona(2024, 2034, freelancer, [
    {
        "event": "Client Meetings",
        "condition": lambda current_date, persona: random.random() < 0.2 and current_date.weekday() < 5,
        # 20% of weekdays
        "start_time": "11:00",
        "end_time": "13:00"  # Perfectly within preferred time
    },
    {
        "event": "Project Work",
        "condition": lambda current_date, persona: current_date.weekday() < 5,  # Weekdays
        "start_time": "14:00",  # Scheduled outside of meeting preference
        "end_time": "18:00"  # Adjusted to reflect 4 hours
    },
    {
        "event": "Professional Development Webinar",
        "condition": lambda current_date, persona: current_date.month % 2 == 0 and current_date.weekday() == 2,
        # Every other month on Tuesdays
        "start_time": "11:00",
        "end_time": "13:00"  # Within preferred time
    },
    {
        "event": "Networking Event",
        "condition": lambda current_date, persona: current_date.month in [1, 6, 12] and current_date.weekday() == 3,
        # January, June, December on Wednesdays
        "start_time": "11:00",
        "end_time": "14:00"  # Slightly extended to accommodate 3 hours
    },
    {
        "event": "Urgent Client Revisions",
        "condition": lambda current_date, persona: random.random() < 0.1 and current_date.weekday() < 5,
        # 10% of weekdays
        "start_time": "15:00",
        "end_time": "17:00"
    },
    {
        "event": "Email and Admin Tasks",
        "condition": lambda current_date, persona: current_date.weekday() < 5,  # Weekdays
        "start_time": "08:00",
        "end_time": "10:00"  # Early morning, suitable for admin
    },
    {
        "event": "Creative Workshop",
        "condition": lambda current_date, persona: current_date.month % 3 == 0 and current_date.weekday() == 5,
        # Every third month on Saturdays
        "start_time": "10:00",
        "end_time": "12:00"  # Weekend workshop
    }
])
hr_manager_meetings = generate_meetings_for_persona(2024, 2034, hr_manager, [
    {
        "event": "Recruitment Interviews",
        "condition": lambda current_date, persona: current_date.weekday() in [0, 3],  # Mondays and Thursdays
        "start_time": "10:00",
        "end_time": "12:00"
    },
    {
        "event": "Team Meetings",
        "condition": lambda current_date, persona: current_date.weekday() == 2,  # Wednesdays
        "start_time": "15:00",
        "end_time": "16:00"
    },
    {
        "event": "HR Training Sessions",
        "condition": lambda current_date, persona: current_date.weekday() == 1,  # Tuesdays
        "start_time": "10:00",
        "end_time": "12:00"
    },
    {
        "event": "Employee Review Meetings",
        "condition": lambda current_date, persona: current_date.weekday() == 3,  # Thursdays
        "start_time": "14:00",
        "end_time": "15:00"
    },
    {
        "event": "Wellness Workshop",
        "condition": lambda current_date, persona: current_date.weekday() == 0 and current_date.day <= 7,
        # First Monday of the month
        "start_time": "15:00",
        "end_time": "17:00"
    }
])
startup_ceo_meetings = generate_meetings_for_persona(2024, 2034, startup_ceo, [
    {
        "event": "Investor Meetings",
        "condition": lambda current_date, persona: current_date.weekday() == 0,  # Mondays
        "start_time": "17:00",
        "end_time": "19:00"
    },
    {
        "event": "All Hands Meetings",
        "condition": lambda current_date, persona: current_date.weekday() == 2,  # Wednesdays
        "start_time": "09:00",
        "end_time": "10:00"
    },
    {
        "event": "Product Strategy Sessions",
        "condition": lambda current_date, persona: current_date.weekday() == 4,  # Fridays
        "start_time": "15:00",
        "end_time": "17:00"
    },
    {
        "event": "Networking Events",
        "condition": lambda current_date, persona: (current_date.weekday() == 5),  # Saturdays
        "start_time": "17:00",
        "end_time": "20:00"
    }
])
retail_manager_meetings = generate_meetings_for_persona(2024, 2034, retail_manager, [
    {
        "event": "Staff Training",
        "condition": lambda current_date, persona: current_date.weekday() in [2, 3],  # Wednesdays and Thursdays
        "start_time": "11:00",
        "end_time": "12:00"
    },
    {
        "event": "Inventory Management Meeting",
        "condition": lambda current_date, persona: current_date.weekday() == 1,  # Tuesdays
        "start_time": "10:00",
        "end_time": "11:00"
    },
    {
        "event": "Marketing Strategy Session",
        "condition": lambda current_date, persona: current_date.weekday() == 4 and current_date.day <= 7,
        # First Friday of the month
        "start_time": "15:00",
        "end_time": "16:00"
    }
])

event_planner_meetings = generate_meetings_for_persona(2024, 2034, event_planner, [
    {
        "event": "Client Consultations",
        "condition": lambda current_date, persona: current_date.weekday() in [1, 4],  # Tuesdays and Fridays
        "start_time": "11:00",
        "end_time": "13:00"
    },
    {
        "event": "Vendor Meetings",
        "condition": lambda current_date, persona: current_date.weekday() == 3,  # Thursdays
        "start_time": "16:00",
        "end_time": "18:00"
    },
    {
        "event": "Event Setup Reviews",
        "condition": lambda current_date, persona: current_date.weekday() == 6,  # Saturdays
        "start_time": "14:00",
        "end_time": "16:00"
    },
    {
        "event": "Post-Event Analysis",
        "condition": lambda current_date, persona: (current_date.weekday() == 0) and (current_date.day <= 7),
        # First Monday of the month
        "start_time": "15:00",
        "end_time": "17:00"
    }
])
construction_manager_meetings = generate_meetings_for_persona(2024, 2034, construction_manager, [
    {
        "event": "Site Safety Meetings",
        "condition": lambda current_date, persona: current_date.weekday() == 0,  # Mondays
        "start_time": "07:00",
        "end_time": "08:00"
    },
    {
        "event": "Weekly Planning Meeting",
        "condition": lambda current_date, persona: current_date.weekday() == 2,  # Wednesdays
        "start_time": "07:00",
        "end_time": "09:00"
    },
    {
        "event": "Contractor Coordination",
        "condition": lambda current_date, persona: current_date.weekday() == 4,  # Fridays
        "start_time": "07:00",
        "end_time": "09:00"
    },
    {
        "event": "Budget Review",
        "condition": lambda current_date, persona: (current_date.weekday() == 0) and (current_date.day <= 7),
        # First Monday of the month
        "start_time": "08:00",
        "end_time": "09:00"
    },
    {
        "event": "Regulatory Compliance Update",
        "condition": lambda current_date, persona: (
                current_date.weekday() == 2 and  # It's Wednesday
                ((current_date.day - 1) // 7 + 1) % 2 == 0  # Week number of the month is even (Bi-weekly)
        ),
        "start_time": "08:00",
        "end_time": "09:00"
    }

])
research_scientist_meetings = generate_meetings_for_persona(2024, 2034, research_scientist, [
    {
        "event": "Lab Group Meetings",
        "condition": lambda current_date, persona: current_date.weekday() == 1,  # Tuesdays
        "start_time": "10:00",
        "end_time": "12:00"
    },
    {
        "event": "Research Seminars",
        "condition": lambda current_date, persona: current_date.weekday() == 3,  # Thursdays
        "start_time": "15:00",
        "end_time": "16:00"
    },
    {
        "event": "Collaborative Research Discussions",
        "condition": lambda current_date, persona: (current_date.weekday() == 4) and (current_date.day <= 7),
        # First Monday of the month
        "start_time": "10:00",
        "end_time": "12:00"
    },
    {
        "event": "Project Review Meetings",
        "condition": lambda current_date, persona: (
                current_date.weekday() == 1 and  # It's Tuesday
                ((current_date.day - 1) // 7 + 1) % 3 == 0  # Every third week of the month
        ),
        "start_time": "10:00",
        "end_time": "12:00"
    },
    {
        "event": "External Conference Calls",
        "condition": lambda current_date, persona: current_date.weekday() == 3,  # Thursdays
        "start_time": "15:00",
        "end_time": "16:00"
    }
])

engineer.fill_schedule(engineer_meetings)
doctor.fill_schedule(doctor_meetings)
teacher.fill_schedule(teacher_meetings)
lawyer.fill_schedule(lawyer_meetings)
developer.fill_schedule(developer_meetings)
consultant.fill_schedule(consultant_meetings)
designer.fill_schedule(designer_meetings)
marketing_specialist.fill_schedule(marketing_specialist_meetings)
fitness_trainer.fill_schedule(fitness_trainer_meetings)
freelancer.fill_schedule(freelancer_meetings)
construction_manager.fill_schedule(construction_manager_meetings)
research_scientist.fill_schedule(research_scientist_meetings)
retail_manager.fill_schedule(retail_manager_meetings)
startup_ceo.fill_schedule(startup_ceo_meetings)
event_planner.fill_schedule(event_planner_meetings)
hr_manager.fill_schedule(hr_manager_meetings)

engineer_all_years_schedule = engineer.calendar.generate_schedule_for_years()
doctor_all_years_schedule = doctor.calendar.generate_schedule_for_years()
teacher_all_years_schedule = teacher.calendar.generate_schedule_for_years()
lawyer_all_years_schedule = lawyer.calendar.generate_schedule_for_years()
developer_all_years_schedule = developer.calendar.generate_schedule_for_years()
consultant_all_years_schedule = consultant.calendar.generate_schedule_for_years()
designer_all_years_schedule = designer.calendar.generate_schedule_for_years()
marketing_specialist_all_years_schedule = marketing_specialist.calendar.generate_schedule_for_years()
fitness_trainer_all_years_schedule = fitness_trainer.calendar.generate_schedule_for_years()
freelancer_all_years_schedule = freelancer.calendar.generate_schedule_for_years()
construction_manager_all_years_schedule = construction_manager.calendar.generate_schedule_for_years()
research_scientist_all_years_schedule = research_scientist.calendar.generate_schedule_for_years()
retail_manager_all_years_schedule = retail_manager.calendar.generate_schedule_for_years()
startup_ceo_all_years_schedule = startup_ceo.calendar.generate_schedule_for_years()
event_planner_all_years_schedule = event_planner.calendar.generate_schedule_for_years()
hr_manager_all_years_schedule = hr_manager.calendar.generate_schedule_for_years()

engineer_matrix = create_schedule_matrix(engineer_all_years_schedule)
doctor_matrix = create_schedule_matrix(doctor_all_years_schedule)
teacher_matrix = create_schedule_matrix(teacher_all_years_schedule)
lawyer_matrix = create_schedule_matrix(lawyer_all_years_schedule)
developer_matrix = create_schedule_matrix(developer_all_years_schedule)
consultant_matrix = create_schedule_matrix(consultant_all_years_schedule)
designer_matrix = create_schedule_matrix(designer_all_years_schedule)
marketing_specialist_matrix = create_schedule_matrix(marketing_specialist_all_years_schedule)
fitness_trainer_matrix = create_schedule_matrix(fitness_trainer_all_years_schedule)
freelancer_matrix = create_schedule_matrix(freelancer_all_years_schedule)
construction_manager_matrix = create_schedule_matrix(construction_manager_all_years_schedule)
research_scientist_matrix = create_schedule_matrix(research_scientist_all_years_schedule)
retail_manager_matrix = create_schedule_matrix(retail_manager_all_years_schedule)
startup_ceo_matrix = create_schedule_matrix(startup_ceo_all_years_schedule)
event_planner_matrix = create_schedule_matrix(event_planner_all_years_schedule)
hr_manager_matrix = create_schedule_matrix(hr_manager_all_years_schedule)

from datetime import date, timedelta


def create_monthly_schedule_matrix(calendar, year, month, start_day=None):
    """
    Generate a schedule matrix for a specific month or a 28-day period starting from a specified day.

    :param calendar: Calendar - The calendar object containing the schedules.
    :param year: int - The year of the desired schedule.
    :param month: int - The month of the desired schedule.
    :param start_day: int (optional) - Day of the month to start the schedule. If None, generates for the whole month.
    :return: dict - A dictionary where each key is a date string and the value is a list of binary values representing booked (1) and free (0) time slots.
    """
    # Determine the start and end dates based on the input
    if start_day:
        start_date = date(year, month, start_day)
        end_date = start_date + timedelta(days=27)  # Create a schedule for the next 28 days
    else:
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)

    current_date = start_date
    schedule_matrix = {}

    # Generate the schedule for the specified period
    while current_date <= end_date:
        day_key = current_date.isoformat()
        day_schedule = calendar.days.get(day_key, {})
        # Convert each day's schedule into a binary format (1 for booked, 0 for free)
        daily_schedule = [1 if day_schedule.get(f"{hour:02d}:{minute:02d}") else 0
                          for hour in range(24) for minute in [0, 15, 30, 45]]
        schedule_matrix[day_key] = daily_schedule
        current_date += timedelta(days=1)

    return schedule_matrix


def combine_persona_schedules(personas, year, month):
    # Calculate the number of days in the month
    if month == 12:
        end_date = date(year + 1, 1, 1)
    else:
        end_date = date(year, month + 1, 1)
    start_date = date(year, month, 1)
    num_days = (end_date - start_date).days

    # Initialize the schedule matrix with zeros
    schedule_matrix = {start_date + timedelta(days=i): [0] * 96 for i in range(num_days)}

    # Process each persona
    for persona in personas:
        persona_month_schedule = persona.calendar.generate_month_schedule(year, month)
        for day_key, daily_schedule in persona_month_schedule.items():
            for time_slot, event in daily_schedule.items():
                if event:  # If there's an event at this time slot
                    day_date = date.fromisoformat(day_key)
                    time_index = time_slot_to_index(time_slot)
                    schedule_matrix[day_date][time_index] = 1

    return schedule_matrix


# Example usage


def time_slot_to_index(time_slot):
    hour, minute = map(int, time_slot.split(':'))
    return hour * 4 + minute // 15


def combine_schedule_matrices(*matrices):
    """
    Combine multiple schedule matrices into a single matrix by adding corresponding time slots.

    :param matrices: tuple of dicts - Multiple schedule matrices to combine.
    :return: dict - A combined schedule matrix where each key is a date string and the value is a list of summed binary values.

    """
    combined_matrix = {}
    for matrix in matrices:
        for date, slots in matrix.items():
            if date not in combined_matrix:
                combined_matrix[date] = slots
            else:
                combined_matrix[date] = [sum(values) for values in zip(combined_matrix[date], slots)]

    return combined_matrix


def accumulate_schedule_matrices(schedule_matrices):
    accumulated_matrix = {}
    print("Number of matrices:", len(schedule_matrices))
    if schedule_matrices:  # Check if the list is not empty
        print("Matrix keys:", schedule_matrices[0].keys())  # This assumes the first element is a dictionary

    for matrix in schedule_matrices:
        for date_str, daily_schedule in matrix.items():
            if date_str not in accumulated_matrix:
                accumulated_matrix[date_str] = np.zeros_like(daily_schedule)
            accumulated_matrix[date_str] += daily_schedule

    return accumulated_matrix


def scale_by_max(matrix):
    max_value = np.max(matrix)
    if max_value == 0:
        print("Max value is zero")
        return matrix  # Avoid division by zero if max_value is zero
    return matrix / max_value


def encode_date(date):
    day_of_week = date.weekday()  # Monday is 0, Sunday is 6
    return [1 if day_of_week == i else 0 for i in range(7)]  # One-hot encoding for day of the week


# Function to encode time information as fractions of the day
def encode_time(hour, minute):
    total_minutes = hour * 60 + minute
    return total_minutes / (24 * 60)  # Normalize to range [0, 1]


def process_labels(data):
    # Convert the input dictionary's keys to a sorted list to ensure chronological order
    sorted_dates = sorted(data.keys())
    matrix = []
    for date in sorted_dates:
        matrix.append(data[date])

    return matrix


# Example usage


def str_to_date(date_str):
    """Converts a date string to a datetime.date object."""
    return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()


def date_to_str(date_obj):
    """Converts a datetime.date object to a date string."""
    return date_obj.strftime('%Y-%m-%d')


def generate_label_for_block(personas, start_date, meeting_duration):
    end_date = start_date + timedelta(days=27)
    total_minutes = int(meeting_duration.total_seconds() / 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    duration_tuple = (hours, minutes)

    best_time = personas[0].find_best_meeting_time_for_all_personas(
        personas, start_date, end_date, duration_tuple)

    return best_time


def generate_four_week_blocks(personas, meeting_duration, num_blocks, start_date, end_date):
    all_blocks = []
    all_labels = []

    for _ in range(num_blocks):
        random_start_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days - 28))
        num_personas_per_block = random.randint(1, len(personas))
        selected_personas = random.sample(personas, num_personas_per_block)

        combined_matrix = None
        for persona in selected_personas:
            persona_schedule = persona.calendar.generate_month_schedule(random_start_date.year, random_start_date.month,
                                                                        random_start_date.day)
            persona_matrix = create_period_schedule_matrix(persona_schedule, random_start_date)

            if combined_matrix is None:
                combined_matrix = persona_matrix
            else:
                combined_matrix[:, :, 2] += persona_matrix[:, :, 2]  # Only sum the availability across personas

        # Scale only the availability data
        combined_matrix[:, :, 2] = scale_by_max(combined_matrix[:, :, 2])

        block = []
        for day_offset in range(28):
            block.append(combined_matrix[day_offset, :, :])  # Append full feature set for each day

        block_array = np.array(block)
        label = generate_label_for_block(selected_personas, random_start_date, meeting_duration)
        label = process_labels(label)
        all_blocks.append(block_array)
        all_labels.append(label)

    return all_blocks, all_labels


def calculate_precision_recall_f1(actual_labels, predicted_labels):
    TP = sum((al == pl) for al, pl in zip(actual_labels, predicted_labels) if al != None)
    FP = sum((al != pl) for al, pl in zip(actual_labels, predicted_labels) if pl != None and al != None)
    FN = sum((al == None) for al in actual_labels)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def pad_label(label, expected_shape):
    """ Pad the label to match the expected shape with zeros. """
    padded_label = []
    if label:
        for line in label:
            padded_line = line + [0] * (expected_shape[1] - len(line))  # Pad each line to the correct length
            padded_label.append(padded_line)
        padded_label += [[0] * expected_shape[1]] * (expected_shape[0] - len(padded_label))  # Pad missing days
    else:
        padded_label = [[0] * expected_shape[1] for _ in
                        range(expected_shape[0])]  # Full padding if label is None or empty
    return padded_label


personas = [doctor, teacher, engineer, lawyer, developer, consultant, designer, marketing_specialist, fitness_trainer,
            freelancer, construction_manager, research_scientist, retail_manager, startup_ceo, event_planner, hr_manager]
meeting_duration = timedelta(hours=1)

all_blocks, all_labels = generate_four_week_blocks(personas, meeting_duration, 1500, date(2024, 1, 1),
                                                   date(2034, 12, 31))

for index, label in enumerate(all_labels):
    if len(label) != len(all_labels[0]):
        print(f"Label at index {index} is of different length: {len(label)} expected {len(all_labels[0])}")

# Pad each label to ensure it has the dimensions (28, 96)
padded_labels = [pad_label(label, (28, 96)) for label in all_labels]
print("Padded labels shape:", np.array(padded_labels).shape)

# Convert the list of padded labels into a NumPy array
all_labels_array = np.array(padded_labels)
all_blocks_array = np.array(all_blocks)

num_features = 96

print("Shape of all_labels_array:", all_labels_array.shape)
print("Shape of all_blocks:", all_blocks_array.shape)
print("Shape of first block:", all_blocks_array[0].shape)
print("Shape of first label:", all_labels_array.shape)

model_file_path = 'my_cnn_model.h5'

import os

model_file_path = 'my_cnn_model.h5'

train_data, validation_data, train_labels, validation_labels = train_test_split(
    all_blocks_array, all_labels_array, test_size=0.2, random_state=42)

model_file_path = 'path_to_your_model.h5'  # Specify the correct path to your model file

# Check if model exists
if os.path.exists(model_file_path):
    # Load the existing model
    scheduler_model = load_model(model_file_path)

    print("Model loaded successfully.")
else:
    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(all_blocks_array, all_labels_array, test_size=0.2,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                      random_state=42)  # 0.25 x 0.8 = 0.2

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)

    # Initialize the Meeting Scheduler Model
    scheduler_model = MeetingSchedulerModel()
    scheduler_model.model.summary()

    # Train the model
    scheduler_model.train(X_train, y_train, X_val, y_val, epochs=50)

    # Save the model
    scheduler_model.model.save(model_file_path)
    scheduler_model.plot_metrics()
    print("Model trained and saved successfully.")


def generate_schedule_block_for_date(start_date, combined_matrix_normalized):
    print("Start Date:", start_date.strftime('%Y-%m-%d'))

    new_schedule_block = []
    for day_offset in range(28):  # 28 days in 4 weeks
        current_date = start_date + timedelta(days=day_offset)
        date_str = current_date.strftime('%Y-%m-%d')
        new_schedule_block.append(combined_matrix_normalized.get(date_str, np.zeros(96)))
    new_schedule_block_array = np.array(new_schedule_block)
    new_schedule_block_reshaped = new_schedule_block_array.reshape((1, 28, 96, 1))
    return new_schedule_block_reshaped


engineerFebruaryCalendar = engineer.calendar
year = 2025
month = 2

day = 1
start_date = date(year, month, day)

teacherFebruaryCalendar = teacher.calendar

lawyerFebruaryCalendar = lawyer.calendar
start_date = date(year, month, day)

teacherSchedule = teacher.calendar.generate_month_schedule(year, month, day)
lawyerSchedule = lawyer.calendar.generate_month_schedule(year, month, day)
engineerSchedule = engineer.calendar.generate_month_schedule(year, month, day)
consultantSchedule = consultant.calendar.generate_month_schedule(year, month, day)
designerSchedule = designer.calendar.generate_month_schedule(year, month, day)
marketingSpecialistSchedule = marketing_specialist.calendar.generate_month_schedule(year, month, day)
import numpy as np


def predict_schedules(engineer_schedule, teacher_schedule, lawyer_schedule):
    # Create period schedule matrices
    engineer_matrix = create_period_schedule_matrix(engineer_schedule, start_date)
    teacher_matrix = create_period_schedule_matrix(teacher_schedule, start_date)
    lawyer_matrix = create_period_schedule_matrix(lawyer_schedule, start_date)

    # Combine the matrices
    combined_availability = engineer_matrix[:, :, 2] + teacher_matrix[:, :, 2] + lawyer_matrix[:, :, 2]

    # Scale only the availability data
    scaled_availability = scale_by_max(combined_availability)

    # Reconstruct the full feature matrix including day and time encodings
    full_feature_matrix = np.zeros((28, 96, 10))  # Adjust the shape based on your model's input
    for day_idx in range(28):
        full_feature_matrix[day_idx, :, :2] = encode_time_of_day()  # Time of day encoding
        full_feature_matrix[day_idx, :, 2] = scaled_availability[day_idx, :]  # Scaled availability
        day_of_week = (start_date.weekday() + day_idx) % 7
        full_feature_matrix[day_idx, :, 3:] = np.tile(encode_day_of_week(day_of_week), (96, 1))  # Day of week encoding

    # Predict using the model
    predictions = scheduler_model.predict(
        full_feature_matrix.reshape(1, 28, 96, 10))  # Ensure the shape matches the model's input

    return predictions


# Example usage
predictions = predict_schedules(consultantSchedule, marketingSpecialistSchedule, lawyerSchedule)
print("Predictions:", predictions.shape)

# Set a threshold for availability
threshold = 0.5

# Boolean mask for available time slots
available_slots = predictions > threshold

# Example: Sum of available time slots per day
available_per_day = np.sum(available_slots, axis=2)  # Sum across time slots

# Find days with the most available slots
most_available_days = np.argsort(-available_per_day.ravel())

# Print the top 5 days with the most available time slots
print("Top 5 days with the most available slots:")
for i in range(5):
    day_index = most_available_days[i]
    print(f"Day {day_index + 1} with {available_per_day.ravel()[day_index]} available slots")

# Additionally, you may want to find the best time slots across all days
flat_indexes = np.argsort(-predictions.ravel())[:30]  # Top 5 time slots
times = [(idx // 96 % 28, idx % 96) for idx in flat_indexes]  # Convert flat index to day and time slot

print("Top 5 time slots for scheduling:")
for day, slot in times:
    print(f"Day {day + 1}, Slot {slot + 1} (Time: {slot * 15 // 60:02d}:{slot * 15 % 60:02d})")

predictions = predictions.reshape(28, 96)

target_data_label = generate_label_for_block([consultant, marketing_specialist, lawyer], start_date=start_date,
                                             meeting_duration=meeting_duration)
target_data_label = process_labels(target_data_label)

target_data_array = np.array(target_data_label, dtype=np.float32)
target_data_label_reshape = target_data_array.reshape(1, 28, 96, 1)  # Match the shape of predictions
target_data_label_reshape = np.squeeze(
    target_data_label_reshape)
# Create a figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
axes[0].imshow(predictions, aspect='auto', interpolation='nearest', cmap='viridis')
axes[0].set_title('Predicted Availability')
axes[0].set_ylabel('Day')
axes[0].set_xlabel('Time Slot')

axes[1].imshow(target_data_label_reshape, aspect='auto', interpolation='nearest', cmap='viridis')
axes[1].set_title('True Availability')
axes[1].set_ylabel('Day')
axes[1].set_xlabel('Time Slot')

# Set tick labels for clarity
time_labels = [f'{(i * 15) // 60:02d}:{(i * 15) % 60:02d}' for i in range(0, 96, 12)]
axes[0].set_xticks(range(0, 96, 12))
axes[0].set_xticklabels(time_labels)
axes[1].set_xticks(range(0, 96, 12))
axes[1].set_xticklabels(time_labels)

plt.tight_layout()
plt.show()

# combined_matrix = accumulate_schedule_matrices([lawyer_monthly_period_schedule, teacher_monthly_period_schedule,
#                                                 engineer_monthly_period_schedule])
# normalized_matrix = scale_by_max(combined_matrix)
#
# target_data_date = date(year, month, day)
# target_data_label = generate_label_for_block([engineer, teacher, lawyer], start_date=target_data_date,
#                                              meeting_duration=meeting_duration)
#
# test_data_array = np.array(normalized_matrix.values(), dtype=np.float32)
# # Reshape the data if necessary (check the expected input shape of your model)
# test_data_array = test_data_array.reshape(1, 28, 96, 1)  # Example reshape, adjust based on your model's input
# print(test_data_array)
# target_data_array = np.array(target_data_label, dtype=np.float32)
# # Ensure target data is reshaped to match prediction output if necessary
# target_data_label_reshape = target_data_array.reshape(1, 28, 96, 1)  # Match the shape of predictions
# target_data_label_reshape = np.squeeze(
#     target_data_label_reshape)  # should now be (28, 96) if your model predicts this shape
#
# # Load your model (this assumes a Keras model; adjust if using scikit-learn)
# model = load_model('path_to_your_model.h5')
#
# # Make predictions
# predictions = model.predict(test_data_array)
#
# # Print the shape of predictions to verify
#
# print("Shape of predictions:", predictions.shape)
#
# # Adjust indexing based on your data
#
# # Squeeze the single-dimensional entry from the array shape
#
# data = np.squeeze(predictions)  # should now be (28, 96)
#
# # Visualizing the data as a heatmap
#
#
# # Calculate MSE and MAE
# mse = mean_squared_error(target_data_label_reshape, data)
# mae = mean_absolute_error(target_data_label_reshape, data)
#
# print("Mean Squared Error:", mse)
# print("Mean Absolute Error:", mae)
#
# rmse = np.sqrt(mean_squared_error(target_data_label_reshape, data))
# print("Root Mean Squared Error:", rmse)
#
# r_squared = r2_score(target_data_label_reshape, data)
# print("R-squared:", r_squared)
#
# explained_variance = explained_variance_score(target_data_label_reshape, data)
# print("Explained Variance Score:", explained_variance)
#
# from sklearn.metrics import precision_score, recall_score, f1_score
#
# # Example for binary classification. Adjust threshold as needed.
# predictions_binary = (data > 0.5).astype(int)
# target_binary = (target_data_label_reshape > 0.5).astype(int)
#
# precision = precision_score(target_binary, predictions_binary, average='macro', zero_division=0)
# recall = recall_score(target_binary, predictions_binary, average='macro', zero_division=0)
# f1 = f1_score(target_binary, predictions_binary, average='macro', zero_division=0)
#
# print("Adjusted Precision (Macro):", precision)
# print("Adjusted Recall (Macro):", recall)
# print("Adjusted F1 Score (Macro):", f1)
#
# from sklearn.metrics import hamming_loss
#
# h_loss = hamming_loss(target_binary, predictions_binary)
# plt.figure(figsize=(20, 10))
#
# # Determine the frequency of the x-ticks (every 3 hours, i.e., every 12 slots if 1 slot is 15 minutes)
# x_ticks_frequency = 12
#
# # Plot predictions
# plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
# plt.imshow(data, cmap='viridis', aspect='auto', interpolation='nearest')
# plt.colorbar()
# plt.title('Heatmap of Model Predictions')
# plt.xlabel('Time Slots (15-min intervals)')
# plt.ylabel('Days')
# plt.xticks(np.arange(0, 96, x_ticks_frequency),
#            [f'{(i * 15) // 60:02d}:{(i * 15) % 60:02d}' for i in range(0, 96, x_ticks_frequency)], rotation=45)
# plt.yticks(np.arange(0, 28, 1), [f'Day {i + 1}' for i in range(28)])
#
# # Plot target data
# plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
# plt.imshow(target_data_label, cmap='viridis', aspect='auto', interpolation='nearest')
# plt.colorbar()
# plt.title('Heatmap of Target Data')
# plt.xlabel('Time Slots (15-min intervals)')
# plt.ylabel('Days')
# plt.xticks(np.arange(0, 96, x_ticks_frequency),
#            [f'{(i * 15) // 60:02d}:{(i * 15) % 60:02d}' for i in range(0, 96, x_ticks_frequency)], rotation=45)
# plt.yticks(np.arange(0, 28, 1), [f'Day {i + 1}' for i in range(28)])
#
# plt.grid(False)
# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()
