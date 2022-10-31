from pymongo import MongoClient
from datetime import datetime

from FisInMa.data_structures import FischerResult, apply_marks, revert_marks


def __get_mongodb_client():
    client = MongoClient('pleyer-ws.fdm.privat', 27017)
    return client


def __get_mongodb_database():
    client = __get_mongodb_client()
    # This should probably for the future be modified to have custom names
    db = client.tsenso_pgaindrik_model_design
    return db


def generate_new_collection(name: str):
    if len(name) < 4:
        raise ValueError("Name too small. Choose a descriptive name with at least 4 characters")
    # Name collection with current time
    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d-%H:%M:%S_")
    collname = dt_string + name
    # Store the collection in the database responsible named tsenso_pgaindrik_model_design
    db = __get_mongodb_database()
    collist = db.list_collection_names()
    if collname in collist:
        raise ValueError("The collection with the name " + collname + " already exists!")
    collection = db[collname]
    print("Created collection with name " + collname)
    return collection


def insert_fischer_dataclasses(fischer_dataclasses, collection):
    coll = get_collection(collection)
    fisses = [f.to_savedict() for f in fischer_dataclasses]
    coll.insert_many(fisses)


def drop_all_collections():
    db = __get_mongodb_database()
    collist = db.list_collection_names()
    for name in collist:
        db.drop_collection(name)


def get_collection(collection):
    db = __get_mongodb_database()
    if type(collection) == str:
        if collection not in db.list_collection_names():
            print("Currently stored collections (names):")
            print(db.list_collection_names())
            raise ValueError("No collection with the name " + collection + " found.")
        else:
            return db[collection]
    else:
        return collection


def list_all_collections():
    db = __get_mongodb_database()
    print(db.list_collection_names())


def get_fischer_results_from_collection(collection):
    coll = get_collection(collection)
    fisses = [[[revert_marks(c[key]) for key in ["criterion", "times", "parameters", "q_arr", "constants", "x0"]]] for c in coll.find()]
    return fisses
