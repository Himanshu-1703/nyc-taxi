from pydantic import BaseModel


class PredictionDataset(BaseModel):
    vendor_id: int
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    pickup_hour: int
    pickup_date: int
    pickup_month: int
    pickup_day: int
    is_weekend: int
    haversine_distance: float
    euclidean_distance: float
    manhattan_distance: float
