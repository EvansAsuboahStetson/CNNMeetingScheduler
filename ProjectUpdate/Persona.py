import random

import numpy as np

from ProjectUpdate.Calendar import Calendar

from datetime import datetime, timedelta, date


class Persona:
    DAY_WEIGHT = 1.0
    TIME_WEIGHT = 1.5
    AVAILABILITY_WEIGHT = 2.0
    WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    FLEXIBILITY_MARGIN = 15 / 60
    HISTORICAL_WEIGHT = 1.0

    def __init__(self, name, availability, preferred_meeting_times, busy_periods, preferred_days,
                 busy_days, historical_meeting_days={},
                 start_year=2024, end_year=2030):
        self.name = name
        self.availability = availability
        self.preferred_meeting_times = preferred_meeting_times
        self.busy_periods = busy_periods
        self.preferred_days = preferred_days
        self.busy_days = busy_days
        self.calendar = Calendar(start_year, end_year)
        self.historical_meeting_days = historical_meeting_days

    def is_available(self, day, start_time, end_time, current_date):
        date_str = current_date.isoformat()

        if day in self.busy_days:
            return False

        # Convert start and end times to decimal hours for availability window check
        start_decimal = start_time[0] + start_time[1] / 60
        end_decimal = end_time[0] + end_time[1] / 60

        availability_start, availability_end = self.availability
        availability_start_decimal = availability_start[0] + availability_start[1] / 60
        availability_end_decimal = availability_end[0] + availability_end[1] / 60

        # Check if time slot is within general availability
        if not (
                availability_start_decimal <= start_decimal < availability_end_decimal and availability_start_decimal < end_decimal <= availability_end_decimal):
            return False

        # Check busy periods
        for busy_start, busy_end in self.busy_periods:
            busy_start_decimal = busy_start[0] + busy_start[1] / 60
            busy_end_decimal = busy_end[0] + busy_end[1] / 60
            if busy_start_decimal < end_decimal and start_decimal < busy_end_decimal:
                return False

        # Check if the time slot is already booked
        day_schedule = self.calendar.get_day_schedule(date_str)
        if day_schedule is None:
            return True  # If no schedule for the day, it means all slots are available

        current_time = datetime(current_date.year, current_date.month, current_date.day, start_time[0], start_time[1])
        end_time_obj = datetime(current_date.year, current_date.month, current_date.day, end_time[0], end_time[1])

        while current_time < end_time_obj:
            time_slot = current_time.strftime("%H:%M")
            if time_slot in day_schedule and day_schedule[time_slot] is not None:
                return False  # Time slot is already booked
            current_time += timedelta(minutes=15)

        return True

    def prefers_meeting(self, time):

        time_decimal = time[0] + time[1] / 60
        for preferred_start, preferred_end in self.preferred_meeting_times:
            preferred_start_decimal = preferred_start[0] + preferred_start[1] / 60
            preferred_end_decimal = preferred_end[0] + preferred_end[1] / 60
            if preferred_start_decimal <= time_decimal < preferred_end_decimal:
                return True
        return False

    def score_meeting(self, current_date, start_time, end_time):
        # Check if current_date is a string and convert it to datetime.date if necessary
        if isinstance(current_date, str):
            current_date = datetime.strptime(current_date, "%Y-%m-%d").date()

        day = current_date.strftime("%A")  # Day name from date
        score = 0
        if self.is_available(day, start_time, end_time, current_date):
            score += self.AVAILABILITY_WEIGHT

        time_score = (self.calculate_time_score(start_time) + self.calculate_time_score(end_time)) / 2
        score += self.TIME_WEIGHT * time_score

        day_score = self.calculate_day_score(day)
        score += self.DAY_WEIGHT * day_score

        historical_score = self.historical_meeting_days.get(day, 0)
        score += self.HISTORICAL_WEIGHT * historical_score

        penalty = self.busy_penalty(start_time, end_time)
        score -= penalty

        return score

    def calculate_day_score(self, day):
        if day in self.preferred_days:
            return 1  # Full score for preferred days
        elif any(adjacent_day(day) in self.preferred_days for adjacent_day in [self.previous_day, self.next_day]):
            return 0.5  # Half score for adjacent preferred days
        return 0

    def previous_day(self, day):
        """
        Returns the name of the day that precedes the given day.

        :param day: str - Name of the current day.
        :return: str - Name of the previous day.
        """
        day_index = self.WEEKDAYS.index(day)  # Find the index of the current day in the list
        previous_day_index = (day_index - 1) % len(self.WEEKDAYS)  # Calculate index of the previous day
        return self.WEEKDAYS[previous_day_index]  # Return the name of the previous day

    def next_day(self, day):
        """
        Returns the name of the day that follows the given day.

        :param day: str - Name of the current day.
        :return: str - Name of the next day.
        """
        day_index = self.WEEKDAYS.index(day)  # Find the index of the current day in the list
        next_day_index = (day_index + 1) % len(self.WEEKDAYS)  # Calculate index of the next day
        return self.WEEKDAYS[next_day_index]

    def calculate_time_score(self, time):
        time_decimal = time[0] + time[1] / 60
        score = 0
        for preferred_start, preferred_end in self.preferred_meeting_times:
            preferred_start_decimal = preferred_start[0] + preferred_start[1] / 60
            preferred_end_decimal = preferred_end[0] + preferred_end[1] / 60
            # Check if within preferred time
            if preferred_start_decimal <= time_decimal < preferred_end_decimal:
                score = 1  # Maximum score for perfect match
            elif abs(time_decimal - preferred_start_decimal) < 1 or abs(time_decimal - preferred_end_decimal) < 1:
                score = 0.5  # Half score for close match
        return score

    def busy_penalty(self, start_time, end_time):
        penalty = 0
        for busy_start, busy_end in self.busy_periods:
            if self.overlaps((start_time, end_time), (busy_start, busy_end)):
                penalty += 4  # Assign a penalty for each overlap with busy periods
        return penalty

    def set_default_schedule(self, default_meetings):
        for start_date, start_time, end_time, event_name in default_meetings:
            year, month, day = map(int, start_date.split('-'))
            self.calendar.book_time_slot(year, month, day, start_time, end_time, event_name)

    def is_available_on_weekends(self, weekday_name):
        is_weekend = weekday_name in ["Saturday", "Sunday"]

        # If it's a weekend, check if the persona works on weekends
        if is_weekend and weekday_name not in self.preferred_days:
            return False
        else:

            return True

    def fill_schedule(self, meetings):
        for meeting in meetings:
            # Convert start_date to date object if necessary
            start_date = meeting['start_date']
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d').date()

            # Handle end_date, convert it to date object if it's a string
            end_date = meeting.get('end_date', start_date)
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

            frequency = meeting.get('frequency', 'once')
            event_name = meeting['event']
            start_time_str = meeting['start_time']
            end_time_str = meeting['end_time']

            start_time = datetime.strptime(start_time_str, "%H:%M").time()
            end_time = datetime.strptime(end_time_str, "%H:%M").time()

            current_date = start_date
            while current_date <= end_date:
                weekday_name = current_date.strftime("%A")

                should_book = False

                if frequency == 'daily':
                    should_book = True
                elif frequency == 'weekly' and current_date.weekday() == start_date.weekday():
                    should_book = True
                elif frequency == 'biweekly' and current_date.weekday() == start_date.weekday() and (
                        (current_date - start_date).days % 14 == 0):
                    should_book = True
                elif frequency == 'monthly' and current_date.day == start_date.day:
                    should_book = True
                elif frequency == 'once':
                    should_book = True

                if should_book and self.is_available(current_date.strftime("%A"),
                                                     (start_time.hour, start_time.minute),
                                                     (end_time.hour, end_time.minute),
                                                     current_date):
                    self.calendar.book_time_slot(current_date.year, current_date.month, current_date.day,
                                                 start_time_str, end_time_str, event_name)
                    if frequency == 'once':
                        break

                current_date += timedelta(days=1)

    def get_schedule(self, start_date, end_date):
        return self.calendar.get_schedule(start_date, end_date)

    @staticmethod
    def normalize_scores(meetings):

        max_score = max(max(times[2] for times in day) for day in meetings.values())
        return {date: [((start, end), score / max_score) for (start, end, score) in times] for date, times in
                meetings.items()}

    @staticmethod
    def generate_labels_for_dates(meetings, start_date, end_date, time_slots_per_day=96):
        normalized_meetings = Persona.normalize_scores(meetings)

        labels = {}
        # Generate labels for each date in the range
        for n in range((end_date - start_date).days + 1):
            single_date = start_date + timedelta(days=n)
            day_label = [0] * time_slots_per_day  # Initialize the day's labels with zeros

            if single_date.isoformat() in normalized_meetings:  # Check if the date has any meetings
                for (start_time, end_time), score in normalized_meetings[single_date.isoformat()]:
                    start_index = (start_time[0] * 60 + start_time[1]) // 15
                    end_index = (end_time[0] * 60 + end_time[1]) // 15
                    for i in range(start_index, end_index):
                        day_label[i] = score  # Set the score for each relevant time slot

            labels[single_date.isoformat()] = day_label

        return labels

    @staticmethod
    def find_best_meeting_time_for_all_personas(personas, start_date, end_date, meeting_duration):
        meeting_times = {}
        current_date = start_date
        while current_date <= end_date:
            day_has_meetings = False
            for persona in personas:
                day_meeting_times = find_meeting_times_for_day(persona, current_date, meeting_duration,
                                                               persona.availability[0], persona.availability[1])
                for time_slot in day_meeting_times:
                    start_time, end_time, score = time_slot
                    if is_mostly_available_at_key_times(personas, current_date, start_time, end_time):
                        meeting_key = (current_date.isoformat(), start_time, end_time)
                        if meeting_key not in meeting_times:
                            meeting_times[meeting_key] = [score]
                        else:
                            meeting_times[meeting_key].append(score)
                        day_has_meetings = True

            # if not day_has_meetings:
            #     print(f"No meetings found on {current_date.isoformat()}.")
            current_date += timedelta(days=1)

        if not meeting_times:
            return {}

        sorted_meeting_times = sorted(meeting_times.items(), key=lambda x: -sum(x[1]))
        clustered_meeting_times = cluster_meeting_times(sorted_meeting_times)
        selected_times = select_best_meetings_per_day(clustered_meeting_times)

        selected_labels = Persona.generate_labels_for_dates(selected_times, start_date, end_date)

        return selected_labels

    @staticmethod
    def book_best_meeting(persona, event_name, meeting_duration, start_range, end_range, start_period,
                          end_period):
        best_times = find_best_meeting_times_over_period(persona, start_period, end_period, meeting_duration,
                                                         start_range,
                                                         end_range)

        for time_slot in best_times:
            day_str, start_time_tuple, end_time_tuple, score = time_slot
            start_time_str = convert_time_tuple_to_string(start_time_tuple)
            end_time_str = convert_time_tuple_to_string(end_time_tuple)
            day_date = datetime.fromisoformat(day_str)
            year, month, day = day_date.year, day_date.month, day_date.day

            if persona.calendar.book_time_slot(year, month, day, start_time_str, end_time_str, event_name):
                return f"Meeting '{event_name}' booked on {day_str} from {start_time_str} to {end_time_str}."
        return "Unable to book a meeting based on the given criteria."

    @staticmethod
    def overlaps(time_range1, time_range2):
        """
        Check if two time ranges overlap.

        :param time_range1: tuple of tuples - First time range.
        :param time_range2: tuple of tuples - Second time range.
        :return: bool - True if they overlap, False otherwise.
        """
        start1, end1 = time_range1
        start2, end2 = time_range2

        start1_decimal = start1[0] + start1[1] / 60
        end1_decimal = end1[0] + end1[1] / 60
        start2_decimal = start2[0] + start2[1] / 60
        end2_decimal = end2[0] + end2[1] / 60

        return max(start1_decimal, start2_decimal) < min(end1_decimal, end2_decimal)


import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO)


def is_mostly_available_at_key_times(personas, current_date, start_time, end_time):
    """
    Checks if the majority of personas are available at the start, midpoint, and end of a proposed meeting time.

    Parameters:
        personas (list): List of persona objects that should have an is_available method.
        current_date (datetime.date): The date of the proposed meeting.
        start_time (tuple): Start time of the meeting as a tuple (hour, minute).
        end_time (tuple): End time of the meeting as a tuple (hour, minute).

    Returns:
        bool: True if at least two out of three key times are mostly available, False otherwise.
    """
    # Calculate total minutes for start and end times
    start_total_minutes = start_time[0] * 60 + start_time[1]
    end_total_minutes = end_time[0] * 60 + end_time[1]

    # Calculate the midpoint time in total minutes
    midpoint_total_minutes = (start_total_minutes + end_total_minutes) // 2

    # Convert midpoint time back to hours and minutes
    midpoint_hour = midpoint_total_minutes // 60
    midpoint_minute = midpoint_total_minutes % 60

    # Calculate availability counts
    count_start_available = sum(
        persona.is_available(current_date.strftime("%A"), start_time, end_time, current_date) for persona in personas)
    count_midpoint_available = sum(
        persona.is_available(current_date.strftime("%A"), (midpoint_hour, midpoint_minute), end_time, current_date) for
        persona in personas)
    count_end_available = sum(
        persona.is_available(current_date.strftime("%A"), end_time, end_time, current_date) for persona in personas)

    # # Logging debug information
    # logging.info(
    #     f"Availability counts - Start: {count_start_available}, Midpoint: {count_midpoint_available}, End: {count_end_available}")

    # Check availability against threshold
    start_available = count_start_available >= len(personas) * 0.5
    midpoint_available = count_midpoint_available >= len(personas) * 0.5
    end_available = count_end_available >= len(personas) * 0.3

    # Evaluate overall availability
    overall_availability = sum([start_available, midpoint_available, end_available]) >= 2
    # logging.info(f"Meeting time validity: {overall_availability}")

    return overall_availability


from datetime import timedelta


def find_best_meeting_times_over_period(persona, start_date, end_date, meeting_duration, start_range, end_range):
    current_date = start_date
    all_meeting_times = []

    while current_date <= end_date:
        # Find meeting times for each day within the specified period
        day_meeting_times = find_meeting_times_for_day(persona, current_date, meeting_duration, start_range, end_range)

        for time_slot in day_meeting_times:
            start_time, end_time, score = time_slot
            # Append meeting times along with the current date, formatted to ISO standard
            all_meeting_times.append((current_date.isoformat(), start_time, end_time, score))

        # Increment the date by one day
        current_date += timedelta(days=1)

    # Optionally sort all meeting times by score if needed (not shown here)
    return all_meeting_times

    all_meeting_times.sort(key=lambda x: x[3], reverse=True)
    return all_meeting_times


def find_meeting_times_for_day(persona, current_date, meeting_duration, start_range, end_range):
    meeting_times = []

    duration_decimal = meeting_duration[0] + meeting_duration[1] / 60
    current_time = start_range

    while current_time[0] + current_time[1] / 60 + duration_decimal <= end_range[0] + end_range[1] / 60:
        end_time = add_time(current_time, meeting_duration)
        score = persona.score_meeting(current_date, current_time, end_time)

        if score > 0:
            meeting_times.append((current_time, end_time, score))

        current_time = add_time(current_time, (0, 15))

    return meeting_times


def scores_are_similar(cluster_scores, current_scores):
    return set(cluster_scores) == set(current_scores)  # Simple check for exact match


def cluster_meeting_times(meeting_times):
    # Sort meetings by date and start time
    if isinstance(meeting_times, dict):
        items = sorted(meeting_times.items(), key=lambda x: (x[0][0], x[0][1]))
    else:
        items = sorted(meeting_times, key=lambda x: (x[0][0], x[0][1]))

    clustered_times = []
    current_cluster = None
    current_scores = []

    for item in items:
        meeting_date, start_time, end_time = item[0]
        scores = item[1]

        if current_cluster is None:
            # Start a new cluster
            current_cluster = (meeting_date, start_time, end_time)
            current_scores = scores
        else:
            current_end_minutes = current_cluster[2][0] * 60 + current_cluster[2][1]
            meeting_start_minutes = start_time[0] * 60 + start_time[1]

            # Check time proximity and score similarity
            if meeting_date == current_cluster[
                0] and meeting_start_minutes <= current_end_minutes + 15 and scores_are_similar(current_scores, scores):
                # Extend the current cluster
                current_cluster = (current_cluster[0], current_cluster[1], end_time)

                current_scores.extend(scores)  # Collecting all scores within the cluster
            else:
                # Calculate the average score for the completed cluster
                average_score = sum(current_scores) / len(current_scores) if current_scores else 0
                clustered_times.append(current_cluster + (average_score,))

                # Start a new cluster
                current_cluster = (meeting_date, start_time, end_time)
                current_scores = scores

    # Don't forget to add the last cluster
    if current_cluster:
        average_score = sum(current_scores) / len(current_scores) if current_scores else 0
        clustered_times.append(current_cluster + (average_score,))
    else:
        print("current_cluster is empty", current_cluster)

    return clustered_times


def select_best_meetings_per_day(clustered_meeting_times, threshold_percentage=10):
    """
    Selects meetings within a certain percentage of the highest score per day.

    Args:
        clustered_meeting_times (list of tuples): A list of tuples representing clustered meeting times.
        threshold_percentage (float): The percentage below the highest score included in the selection.

    Returns:
        dict: A dictionary with the date as keys and a list of meeting times that are within the score threshold.
    """
    from collections import defaultdict

    best_meetings_per_day = defaultdict(list)
    meetings_by_day = defaultdict(list)

    for meeting in clustered_meeting_times:
        day = meeting[0]
        start_time = meeting[1]
        end_time = meeting[2]
        score = meeting[3]
        meetings_by_day[day].append((start_time, end_time, score))

    for day, meetings in meetings_by_day.items():
        best_score = max(meeting[2] for meeting in meetings)
        score_threshold = best_score - (best_score * threshold_percentage / 100)

        for start_time, end_time, score in meetings:
            if score >= score_threshold:
                best_meetings_per_day[day].append((start_time, end_time, score))

    return best_meetings_per_day


def add_time(time, duration):
    hours, minutes = time
    d_hours, d_minutes = duration
    new_minutes = minutes + d_minutes
    new_hours = hours + d_hours + new_minutes // 60
    new_minutes %= 60
    return new_hours, new_minutes


def convert_time_tuple_to_string(time_tuple):
    """
    Convert a time tuple (hour, minute) to a string in 'HH:MM' format.
    :param time_tuple: tuple - Time tuple in the form (hour, minute).
    :return: str - Time in 'HH:MM' format.
    """
    hours, minutes = time_tuple
    return f"{hours:02d}:{minutes:02d}"

# Create personas
