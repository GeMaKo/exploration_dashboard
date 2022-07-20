import abc

from .base import Dataset


class DatasetManager:
    def __init__(self) -> None:
        self.datasets = self._get_datasets()

    @staticmethod
    def _get_datasets() -> dict[str, abc.ABCMeta]:
        sub_cls = Dataset.__subclasses__()
        sub_cls_names = tuple(cls.name for cls in sub_cls)
        return {n: cls for n, cls in zip(sub_cls_names, sub_cls)}

    def init_dataset(self, name: str) -> abc.ABCMeta:
        cls = self.datasets.get(name, object)
        return cls()
