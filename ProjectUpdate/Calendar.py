from datetime import datetime, date, timedelta  # Import the datetime class explicitly

import numpy as np

import matplotlib.pyplot as plt
import numpy as np


class Calendar:
    def __init__(self, start_year, end_year):
        self.start_year = start_year
        self.end_year = end_year
        self.days = self.generate_days(start_year, end_year)

    def get_day_schedule(self, date):

        """
        Get the schedule for a specific day.

        :param date: str - Date in ISO format (yyyy-mm-dd).
        :return: dict - Dictionary of time slots and events for the given day, or None if the day is not found.

        """
        date_str = date if isinstance(date, str) else date.isoformat()
        return self.days.get(date_str)

    def generate_month_schedule(self, year, month, start_day=None):
        """
        Generate the schedule for a specific month or a 28-day period starting from a specified day.

        :param year: int - Year of the schedule.
        :param month: int - Month of the schedule (1-12).
        :param start_day: int (optional) - Day of the month to start the schedule. If None, generates for the whole month.
        :return: dict - Dictionary representing the schedule for each day of the specified period.
        """
        # Determine the start and end dates based on the input
        if start_day:
            start_date = date(year, month, start_day)
            end_date = start_date + timedelta(days=27)  # 28 days including the start day
        else:
            start_date = date(year, month, 1)
            if month == 12:
                end_date = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(year, month + 1, 1) - timedelta(days=1)

        current_date = start_date
        schedule = {}

        while current_date <= end_date:
            day_key = current_date.isoformat()
            schedule[day_key] = self.days.get(day_key, {})
            current_date += timedelta(days=1)

        return schedule


    def generate_year_schedule(self, year):
        year_schedule = {}
        for month in range(1, 13):
            month_schedule = self.generate_month_schedule(year, month)
            year_schedule[month] = month_schedule
        return year_schedule

    def generate_schedule_for_years(self):
        all_years_schedule = {}
        for year in range(self.start_year, self.end_year + 1):
            all_years_schedule[year] = self.generate_year_schedule(year)
        return all_years_schedule

    @staticmethod
    def generate_days(start_year, end_year):
        days = {}
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                for day in range(1, 32):
                    try:
                        # Use date directly to create a date object
                        date_obj = date(year, month, day)
                    except ValueError:
                        continue
                    day_key = date_obj.isoformat()
                    days[day_key] = {time: None for time in Calendar.generate_time_slots()}
        return days

    @staticmethod
    def generate_time_slots():
        """
        Generate 15-minute time slots for a day.

        :return: list - List of time slots.
        """
        times = []
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                times.append(f"{hour:02d}:{minute:02d}")
        return times
    

    @staticmethod
    def plot_month_schedule(schedule, title):
        print("Schedule to plot:", schedule.shape)  # Display the shape of the schedule array

        # Assuming 'schedule' is an array with dimensions [days, time_slots]
        # Example dimensions [28, 96] where 28 represents days and 96 time slots per day.

        # Plotting
        fig, ax = plt.subplots(figsize=(15, 10))  # Adjust figure size as needed
        cax = ax.matshow(schedule, cmap='viridis', aspect='auto')  # Use 'auto' aspect to stretch across the axis
        plt.colorbar(cax)

        # Assuming time slots are in increments that could be labeled based on index
        time_slots = [f"{(i//4):02d}:{(i%4)*15:02d}" for i in range(schedule.shape[1])]
        days = [f"Day {i+1}" for i in range(schedule.shape[0])]

        # Format x-ticks for exact time slots
        plt.xticks(range(len(time_slots)), time_slots, rotation=90)  # Rotate labels for better readability

        # Format y-ticks for days
        plt.yticks(range(len(days)), days)

        plt.ylabel('Day')
        plt.xlabel('Time Slot')
        plt.title(title)

        ax.grid(False)
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()


    def book_time_slot(self, year, month, day, start_time, end_time, event):
        """
        Book a time slot with an event.

        :param year: int - Year of the date.
        :param month: int - Month of the date.
        :param day: int - Day of the date.
        :param start_time: str - Start time of the event in 24-hour format (HH:MM).
        :param end_time: str - End time of the event in 24-hour format (HH:MM).
        :param event: str - Event description.
        :return: bool - True if booked successfully, False otherwise.
        """
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        start_hour, start_minute = map(int, start_time.split(':'))
        end_hour, end_minute = map(int, end_time.split(':'))
        current_time = datetime(year, month, day, start_hour, start_minute)
        end_datetime = datetime(year, month, day, end_hour, end_minute)

        while current_time < end_datetime:
            time_slot = current_time.strftime("%H:%M")
            if not self.is_time_slot_available(date_str, time_slot):
                return False  # Time slot is already booked, can't book the event
            current_time += timedelta(minutes=15)

        # If all time slots are available, book the event
        current_time = datetime(year, month, day, start_hour, start_minute)
        while current_time < end_datetime:
            time_slot = current_time.strftime("%H:%M")
            self.days[date_str][time_slot] = event
            current_time += timedelta(minutes=15)

        return True

    def is_time_slot_available(self, date, time):
        """
        Check if a specific time slot is available.

        :param date: str - Date in ISO format (yyyy-mm-dd).
        :param time: str - Time in 24-hour format (HH:MM).
        :return: bool - True if available, False otherwise.
        """
        return self.days.get(date, {}).get(time, None) is None


def convert_time_tuple_to_string(time_tuple):
    """
    Convert a time tuple (hour, minute) to a string in 'HH:MM' format.

    :param time_tuple: tuple - Time tuple in the form (hour, minute).
    :return: str - Time in 'HH:MM' format.
    """
    hour, minute = time_tuple
    return f"{hour:02d}:{minute:02d}"

