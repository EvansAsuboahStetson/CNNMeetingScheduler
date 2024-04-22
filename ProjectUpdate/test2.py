import random
from datetime import date, timedelta
from ProjectUpdate.Persona import Persona
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, \
    precision_score, recall_score, f1_score, hamming_loss
from meetingConfig import get_lawyer_meetings, get_doctor_meetings, get_engineer_meetings, get_designer_meetings, \
    get_consultant_meetings, get_marketing_specialist_meetings, get_teacher_meetings, get_fitness_trainer_meetings, \
    get_freelancer_meetings, get_developer_meetings, \
    get_hr_manager_meetings, get_construction_manager_meetings, get_retail_manager_meetings, get_startup_ceo_meetings, \
    get_research_scientist_meetings, get_event_planner_meetings

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss


class ScheduleManager:
    def __init__(self, personas):
        self.personas = {persona.name: persona for persona in personas}

    def setup_meetings(self):
        doctor_meetings = get_doctor_meetings()
        teacher_meetings = get_teacher_meetings()
        engineer_meetings = get_engineer_meetings()
        lawyer_meetings = get_lawyer_meetings()
        designer_meetings = get_designer_meetings()
        consultant_meetings = get_consultant_meetings()
        marketing_specialist_meetings = get_marketing_specialist_meetings()
        fitness_trainer_meetings = get_fitness_trainer_meetings()
        freelancer_meetings = get_freelancer_meetings()
        developer_meetings = get_developer_meetings()
        hr_manager_meetings = get_hr_manager_meetings()
        construction_manager_meetings = get_construction_manager_meetings()
        retail_manager_meetings = get_retail_manager_meetings()
        startup_ceo_meetings = get_startup_ceo_meetings()
        event_planner_meetings = get_event_planner_meetings()
        research_scientist_meetings = get_research_scientist_meetings()

        return doctor_meetings, teacher_meetings, engineer_meetings, lawyer_meetings, designer_meetings, consultant_meetings, marketing_specialist_meetings, fitness_trainer_meetings, freelancer_meetings, developer_meetings, hr_manager_meetings, construction_manager_meetings, retail_manager_meetings, startup_ceo_meetings, event_planner_meetings, research_scientist_meetings

    def generate_meetings_for_persona(self, start_year, end_year, persona_name, meeting_types):
        persona = self.personas[persona_name]
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
                        start_time, end_time = self.generate_random_time_slot_within_availability(
                            persona.availability, duration_hours)

                    meetings.append({
                        "start_date": current_date,
                        "frequency": "once",
                        "event": meeting_type['event'],
                        "start_time": start_time,
                        "end_time": end_time
                    })
            if random.random() < 0.10:
                start_time, end_time = self.generate_random_time_slot_within_availability(persona.availability)
                meetings.append({
                    "start_date": current_date,
                    "frequency": "once",
                    "event": "Random Meeting",
                    "start_time": start_time,
                    "end_time": end_time
                })

            current_date += timedelta(days=1)

        return meetings

    def create_monthly_schedule_matrix(self, calendar, year, month, start_day=None):
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

    def generate_random_time_slot_within_availability(self, availability, meeting_duration_hours=1):
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

    def create_schedule_matrix(self, persona_calendar):
        schedule_matrix = {}
        for year in persona_calendar.keys():
            for month_str in persona_calendar[year].keys():
                for date_str, day_schedule in persona_calendar[year][month_str].items():
                    daily_schedule = [1 if day_schedule[time_slot] else 0 for time_slot in day_schedule]
                    schedule_matrix[date_str] = daily_schedule
        return schedule_matrix

    def generate_label_for_block(self, personas, start_date, meeting_duration):
        """
        Generate optimal meeting time label for a block of days.
        """
        print("Generating label for block of days...")
        end_date = start_date + timedelta(days=27)
        total_minutes = int(meeting_duration.total_seconds() / 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        duration_tuple = (hours, minutes)

        best_time = personas[0].find_best_meeting_time_for_all_personas(
            personas, start_date, end_date, duration_tuple)
        print("Best time:", best_time)
        return best_time

    @staticmethod
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


class DataAnalysis:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def combine_schedule_matrices(self, *matrices):
        """
        Combine multiple schedule matrices into a single matrix by adding corresponding time slots.
        """
        combined_matrix = {}
        for matrix in matrices:
            for date, slots in matrix.items():
                if date not in combined_matrix:
                    combined_matrix[date] = slots
                else:
                    combined_matrix[date] = [sum(values) for values in zip(combined_matrix[date], slots)]
        return combined_matrix

    def accumulate_schedule_matrices(self, schedule_matrices):
        print("Accumulating schedule matrices...", end=" ")
        print(schedule_matrices, end=" ")
        print("Number of matrices:", len(schedule_matrices))

        # Initialize the accumulated matrix only if there is at least one matrix to process
        if schedule_matrices:
            accumulated_matrix = np.zeros_like(schedule_matrices[0])

            for matrix in schedule_matrices:
                accumulated_matrix += matrix

        return accumulated_matrix

    def scale_by_max(self, matrix):
        max_value = np.max(matrix)
        if max_value == 0:
            print("Max value is zero")
            return matrix  # Avoid division by zero if max_value is zero
        print("Max value:", max_value)
        return matrix / max_value

    def process_labels(self, data):
        # Convert the input dictionary's keys to a sorted list to ensure chronological order
        sorted_dates = sorted(data.keys())
        matrix = []
        for date in sorted_dates:
            matrix.append(data[date])

        return matrix

    def prepare_data(self, matrix):
        print("Preparing data...")
        print("Matrix shape:", matrix.shape)  # Display the shape of the input matrix

        # Ensure the matrix has the expected shape or resize it accordingly
        if matrix.shape != (28, 96):
            raise ValueError("The matrix does not have the expected shape of (28, 96).")

        # Reshape the matrix as needed for further processing
        data_array = matrix.reshape(1, 28, 96, 1)  # Reshape for batch size of 1

        return data_array

    def predict(self, data_array):
        return self.model.predict(data_array)

    from sklearn.metrics import f1_score

    def evaluate_predictions(self, predictions, true_labels):
        predictions = np.squeeze(predictions)

        # Import right inside the function to avoid conflicts
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, \
            precision_score, recall_score, f1_score, hamming_loss

        mse = mean_squared_error(true_labels, predictions)
        mae = mean_absolute_error(true_labels, predictions)
        rmse = np.sqrt(mse)
        r_squared = r2_score(true_labels, predictions)
        explained_variance = explained_variance_score(true_labels, predictions)

        binary_predictions = (predictions > 0.5).astype(int)
        binary_true_labels = (true_labels > 0.5).astype(int)

        precision = precision_score(binary_true_labels, binary_predictions, average='macro', zero_division=0)
        recall = recall_score(binary_true_labels, binary_predictions, average='macro', zero_division=0)
        f1 = f1_score(binary_true_labels, binary_predictions, average='macro', zero_division=0)
        h_loss = hamming_loss(binary_true_labels, binary_predictions)

        return {
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "R-squared": r_squared,
            "Explained Variance": explained_variance,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Hamming Loss": h_loss
        }

    def plot_results(self, predictions, true_labels):
        predictions = np.squeeze(predictions)
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(predictions, cmap='viridis', aspect='auto', interpolation='nearest')
        plt.colorbar()
        plt.title('Heatmap of Model Predictions')
        plt.xlabel('Time Slots (15-min intervals)')
        plt.ylabel('Days')
        plt.xticks(np.arange(0, 96, 12), [f'{(i * 15) // 60:02d}:{(i * 15) % 60:02d}' for i in range(0, 96, 12)],
                   rotation=45)
        plt.yticks(np.arange(0, 28, 1), [f'Day {i + 1}' for i in range(28)])

        plt.subplot(1, 2, 2)
        plt.imshow(true_labels, cmap='viridis', aspect='auto', interpolation='nearest')
        plt.colorbar()
        plt.title('Heatmap of Target Data')
        plt.xlabel('Time Slots (15-min intervals)')
        plt.ylabel('Days')
        plt.xticks(np.arange(0, 96, 12), [f'{(i * 15) // 60:02d}:{(i * 15) % 60:02d}' for i in range(0, 96, 12)],
                   rotation=45)
        plt.yticks(np.arange(0, 28, 1), [f'Day {i + 1}' for i in range(28)])

        plt.tight_layout()
        plt.show()


doctor = Persona(
    name="Doctor",
    availability=((9, 0), (17, 0)),  # Available from 9:00 to 17:00
    preferred_meeting_times=[((10, 0), (12, 0)), ((15, 0), (17, 0))],
    # Prefers meetings between 10:00-12:00 and 15:00-17:00
    busy_periods=[((12, 0), (13, 0))],  # Busy during 12:00-13:00 (lunch break)
    preferred_days=["Monday", "Wednesday", "Thursday", "Friday"],
    busy_days=["Tuesday"],
    start_year=2023,
    end_year=2042
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
    end_year=2042
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
    end_year=2042
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
    end_year=2042
)

developer = Persona(
    name="Software Developer",
    availability=((9, 0), (17, 0)),  # Typical office hours
    preferred_meeting_times=[((10, 0), (11, 0)), ((15, 0), (16, 0))],  # Mid-morning and late afternoon
    busy_periods=[],  # Flexible breaks
    preferred_days=["Monday", "Wednesday", "Friday", "Saturday"],
    busy_days=["Tuesday", "Thursday"],
    start_year=2023,
    end_year=2042
)

consultant = Persona(
    name="Consultant",
    availability=((10, 0), (18, 0)),  # Available from 10:00 to 18:00
    preferred_meeting_times=[((11, 0), (13, 0)), ((15, 0), (17, 0))],  # Morning and late afternoon
    busy_periods=[((13, 0), (14, 0))],  # Lunch break at 13:00-14:00
    preferred_days=["Monday", "Wednesday", "Friday"],
    busy_days=["Saturday", "Sunday", "Thursday"],
    start_year=2023,
    end_year=2042
)
designer = Persona(
    name="Designer",
    availability=((9, 30), (17, 30)),  # Available from 9:30 to 17:30
    preferred_meeting_times=[((10, 0), (12, 0)), ((14, 0), (16, 0))],  # Morning and early afternoon
    busy_periods=[],  # No specific busy periods
    preferred_days=["Monday", "Wednesday", "Friday"],
    busy_days=["Tuesday", "Thursday", "Saturday", "Sunday"],
    start_year=2023,
    end_year=2042
)

marketing_specialist = Persona(
    name="Marketing Specialist",
    availability=((8, 30), (17, 30)),  # Available from 8:30 to 17:30
    preferred_meeting_times=[((9, 0), (11, 0)), ((14, 0), (16, 0))],  # Late morning and mid-afternoon
    busy_periods=[((12, 0), (13, 0))],  # Lunch break at 12:00-13:00
    preferred_days=["Monday", "Wednesday", "Thursday", "Friday"],
    busy_days=["Saturday", "Sunday"],
    start_year=2023,
    end_year=2042
)
fitness_trainer = Persona(
    name="Fitness Trainer",
    availability=((6, 0), (20, 0)),  # Early mornings to evenings
    preferred_meeting_times=[((6, 0), (9, 0)), ((17, 0), (20, 0))],  # Prefer early mornings and evenings for sessions
    busy_periods=[((13, 0), (15, 0))],  # Midday break
    preferred_days=["Monday", "Tuesday", "Thursday", "Friday"],
    busy_days=["Saturday", "Sunday", "Wednesday"],
    start_year=2023,
    end_year=2042
)

freelancer = Persona(
    name="Freelancer",
    availability=((8, 0), (22, 0)),  # Flexible working hours
    preferred_meeting_times=[((11, 0), (13, 0))],  # Prefers late morning meetings
    busy_periods=[],  # Varies based on project
    preferred_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    busy_days=[],  # Can work any day depending on project
    start_year=2023,
    end_year=2042
)
construction_manager = Persona(
    name="Construction Project Manager",
    availability=((6, 0), (15, 0)),  # Early start to avoid late hours
    preferred_meeting_times=[((7, 0), (9, 0))],  # Early meetings before site work begins
    busy_periods=[((12, 0), (13, 0))],  # Short lunch break
    preferred_days=["Monday", "Wednesday", "Friday"],
    busy_days=["Saturday"],  # Often on site on weekends
    start_year=2023,
    end_year=2042
)
research_scientist = Persona(
    name="Research Scientist",
    availability=((8, 0), (17, 0)),  # Typical academic hours
    preferred_meeting_times=[((10, 0), (12, 0)), ((15, 0), (16, 0))],  # Mid-morning and mid-afternoon
    busy_periods=[],  # Flexibility depending on experiments
    preferred_days=["Monday", "Tuesday", "Thursday"],
    busy_days=[],  # Works any weekday depending on research needs
    start_year=2023,
    end_year=2042
)

retail_manager = Persona(
    name="Retail Store Manager",
    availability=((10, 0), (18, 0)),  # Standard retail hours
    preferred_meeting_times=[((11, 0), (12, 0))],  # Late morning before the rush
    busy_periods=[((13, 0), (14, 0))],  # Lunchtime rush
    preferred_days=["Tuesday", "Wednesday", "Thursday"],
    busy_days=["Saturday", "Sunday"],  # Peak shopping days
    start_year=2023,
    end_year=2042
)

startup_ceo = Persona(
    name="Startup CEO",
    availability=((8, 0), (20, 0)),  # Long hours
    preferred_meeting_times=[((9, 0), (10, 0)), ((17, 0), (19, 0))],  # Early meetings or late evening wrap-ups
    busy_periods=[((12, 0), (13, 0))],  # Quick lunch breaks
    preferred_days=["Monday", "Wednesday", "Friday"],
    busy_days=["Saturday"],  # Often busy with networking on weekends
    start_year=2023,
    end_year=2042
)

event_planner = Persona(
    name="Event Planner",
    availability=((10, 0), (19, 0)),  # Later start, works evenings
    preferred_meeting_times=[((11, 0), (13, 0)), ((16, 0), (18, 0))],  # Midday and late afternoon
    busy_periods=[],  # Very flexible but often out on site visits
    preferred_days=["Tuesday", "Thursday", "Saturday"],
    busy_days=["Sunday"],  # Major event days often fall on weekends
    start_year=2023,
    end_year=2042
)

hr_manager = Persona(
    name="HR Manager",
    availability=((9, 0), (17, 0)),  # Office hours
    preferred_meeting_times=[((10, 0), (11, 0)), ((15, 0), (16, 0))],  # Late morning and late afternoon
    busy_periods=[((12, 0), (13, 0))],  # Lunch break
    preferred_days=["Monday", "Tuesday", "Thursday"],
    busy_days=["Wednesday", "Friday"],  # Busy with workshops and training sessions
    start_year=2023,
    end_year=2042
)

# Adjust generate_meetings_for_persona and events as necessary


personas = [doctor, lawyer, engineer, teacher, designer, consultant, marketing_specialist, fitness_trainer, freelancer,
            developer, construction_manager, research_scientist, retail_manager, startup_ceo, event_planner, hr_manager]
schedule_manager = ScheduleManager(personas)

engineer_calendar = engineer.calendar
teacher_calendar = teacher.calendar
lawyer_calendar = lawyer.calendar
doctor_calendar = doctor.calendar
designer_calendar = designer.calendar
consultant_calendar = consultant.calendar
marketing_specialist_calendar = marketing_specialist.calendar
fitness_trainer_calendar = fitness_trainer.calendar
freelancer_calendar = freelancer.calendar
developer_calendar = developer.calendar
construction_manager_calendar = construction_manager.calendar
research_scientist_calendar = research_scientist.calendar
retail_manager_calendar = retail_manager.calendar
startup_ceo_calendar = startup_ceo.calendar
event_planner_calendar = event_planner.calendar
hr_manager_calendar = hr_manager.calendar

doctor_meetings, teacher_meetings, engineer_meetings, lawyer_meetings, designer_meetings, consultant_meetings, marketing_specialist_meetings, fitness_trainer_meetings, freelancer_meetings, developer_meetings, hr_manager_meetings, construction_manager_meetings, retail_manager_meetings, startup_ceo_meetings, event_planner_meetings, research_scientist_meetings = schedule_manager.setup_meetings()

engineer_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, engineer.name, engineer_meetings)
teacher_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, teacher.name, teacher_meetings)
lawyer_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, lawyer.name, lawyer_meetings)
doctor_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, doctor.name, doctor_meetings)
designer_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, designer.name, designer_meetings)
consultant_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, consultant.name, consultant_meetings)
marketing_specialist_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, marketing_specialist.name,
                                                                               marketing_specialist_meetings)
fitness_trainer_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, fitness_trainer.name,
                                                                          fitness_trainer_meetings)
freelancer_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, freelancer.name, freelancer_meetings)
developer_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, developer.name, developer_meetings)
hr_manager_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, hr_manager.name, hr_manager_meetings)
construction_manager_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, construction_manager.name,
                                                                               construction_manager_meetings)
retail_manager_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, retail_manager.name,
                                                                         retail_manager_meetings)
startup_ceo_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, startup_ceo.name,
                                                                      startup_ceo_meetings)
event_planner_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, event_planner.name,
                                                                        event_planner_meetings)
research_scientist_meetings = schedule_manager.generate_meetings_for_persona(2035, 2039, research_scientist.name,
                                                                             research_scientist_meetings)

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
hr_manager.fill_schedule(hr_manager_meetings)
construction_manager.fill_schedule(construction_manager_meetings)
retail_manager.fill_schedule(retail_manager_meetings)
startup_ceo.fill_schedule(startup_ceo_meetings)
event_planner.fill_schedule(event_planner_meetings)
research_scientist.fill_schedule(research_scientist_meetings)

# Assuming you have imported the necessary personas and meeting data
analysis = DataAnalysis('path_to_your_model.h5')

year = 2038
month = 1

day = 1
start_date = date(year, month, day)

teacherSchedule = teacher.calendar.generate_month_schedule(year, month, day)
lawyerSchedule = lawyer.calendar.generate_month_schedule(year, month, day)
engineerSchedule = engineer.calendar.generate_month_schedule(year, month, day)
doctorSchedule = doctor.calendar.generate_month_schedule(year, month, day)
developerSchedule = developer.calendar.generate_month_schedule(year, month, day)
consultantSchedule = consultant.calendar.generate_month_schedule(year, month, day)
designerSchedule = designer.calendar.generate_month_schedule(year, month, day)
marketing_specialistSchedule = marketing_specialist.calendar.generate_month_schedule(year, month, day)
fitness_trainerSchedule = fitness_trainer.calendar.generate_month_schedule(year, day, month)
freelancerSchedule = freelancer.calendar.generate_month_schedule(year, month, day)
hr_managerSchedule = hr_manager.calendar.generate_month_schedule(year, month, day)
construction_managerSchedule = construction_manager.calendar.generate_month_schedule(year, month, day)
retail_managerSchedule = retail_manager.calendar.generate_month_schedule(year, month, day)
startup_ceoSchedule = startup_ceo.calendar.generate_month_schedule(year, month, day)
event_plannerSchedule = event_planner.calendar.generate_month_schedule(year, month, day)
research_scientistSchedule = research_scientist.calendar.generate_month_schedule(year, month, day)


def scale_by_max(matrix):
    max_value = np.max(matrix)
    if max_value == 0:
        print("Max value is zero")
        return matrix  # Avoid division by zero if max_value is zero
    return matrix / max_value


def encode_day_of_week(day):
    # Returns a one-hot encoded vector for the day of the week
    return np.eye(7)[day]


def encode_time_of_day():
    # Returns a cyclic encoding for each time slot of the day
    times_per_day = 96  # Assuming there are 96 time slots per day (15 minutes each)
    return np.array(
        [(np.sin(2 * np.pi * i / times_per_day), np.cos(2 * np.pi * i / times_per_day)) for i in range(times_per_day)])


def predict_schedules(schedule_list, start_date):
    # Create period schedule matrices for each schedule in the list
    matrices = []
    for schedule in schedule_list:
        matrix = schedule_manager.create_period_schedule_matrix(schedule, start_date)
        matrices.append(matrix)

    # Combine availability from all schedules
    combined_availability = sum(matrix[:, :, 2] for matrix in matrices)

    # Scale the combined availability data
    scaled_availability = scale_by_max(combined_availability)

    # Reconstruct the full feature matrix including day and time encodings
    full_feature_matrix = np.zeros((28, 96, 10))  # Adjust the shape based on your model's input
    for day_idx in range(28):
        full_feature_matrix[day_idx, :, :2] = encode_time_of_day()  # Time of day encoding
        full_feature_matrix[day_idx, :, 2] = scaled_availability[day_idx, :]  # Scaled availability
        day_of_week = (start_date.weekday() + day_idx) % 7
        full_feature_matrix[day_idx, :, 3:] = np.tile(encode_day_of_week(day_of_week), (96, 1))

    model = load_model('path_to_your_model.h5')

    # Reshape the full feature matrix to match the model's input shape and make predictions
    predictions = model.predict(full_feature_matrix.reshape(1, 28, 96, 10))

    return predictions

array_of_personas_for_prediction = [fitness_trainer, startup_ceo]
array_of_Schedules_for_prediction = [fitness_trainerSchedule, startup_ceoSchedule]
# Example usage
predictions = predict_schedules(array_of_Schedules_for_prediction, start_date)
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
meeting_duration = timedelta(hours=1)  # Example duration, adjust as necessary

target_data_label = schedule_manager.generate_label_for_block(array_of_personas_for_prediction,
                                                              start_date=start_date,
                                                              meeting_duration=meeting_duration)
target_data_label = analysis.process_labels(target_data_label)

target_data_array = np.array(target_data_label, dtype=np.float32)
target_data_label_reshape = target_data_array.reshape(1, 28, 96, 1)  # Match the shape of predictions
target_data_label_reshape = np.squeeze(
    target_data_label_reshape)

# Plotting setup
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

# Plot predicted availability
axes[0].imshow(predictions, aspect='auto', interpolation='nearest', cmap='viridis')
axes[0].set_title('Predicted Availability')
axes[0].set_ylabel('Day')
axes[0].set_xlabel('Time Slot')

# Plot true availability
axes[1].imshow(target_data_array, aspect='auto', interpolation='nearest', cmap='viridis')
axes[1].set_title('True Availability')
axes[1].set_ylabel('Day')
axes[1].set_xlabel('Time Slot')

# Set tick labels for clarity on the x-axis (Time Slots)
time_labels = [f'{(i * 15) // 60:02d}:{(i * 15) % 60:02d}' for i in range(0, 96, 8)]  # Every 8 slots for 2 hours
axes[0].set_xticks(range(0, 96, 8))
axes[0].set_xticklabels(time_labels)
axes[1].set_xticks(range(0, 96, 8))
axes[1].set_xticklabels(time_labels)

# Set tick labels for clarity on the y-axis (Days)
day_labels = [f'Day {i + 1}' for i in range(0, 28, 2)]  # Every 2 days
axes[0].set_yticks(range(0, 28, 2))
axes[0].set_yticklabels(day_labels)
axes[1].set_yticks(range(0, 28, 2))
axes[1].set_yticklabels(day_labels)

plt.tight_layout()
plt.show()

print("Evaluating predictions...")
print(predictions.shape, target_data_array.shape)

f1_score = analysis.evaluate_predictions(predictions, target_data_array)["F1 Score"]
mse = analysis.evaluate_predictions(predictions, target_data_array)["MSE"]
mae = analysis.evaluate_predictions(predictions, target_data_array)["MAE"]
rmse = analysis.evaluate_predictions(predictions, target_data_array)["RMSE"]
r_squared = analysis.evaluate_predictions(predictions, target_data_array)["R-squared"]
explained_variance = analysis.evaluate_predictions(predictions, target_data_array)["Explained Variance"]
precision = analysis.evaluate_predictions(predictions, target_data_array)["Precision"]

print("F1 Score:", f1_score)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r_squared)
print("Explained Variance:", explained_variance)
print("Precision:", precision)
