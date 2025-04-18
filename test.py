from ingest import BookEncoder

encoder = BookEncoder()
encoder.encode_directory("sources", "*.pdf")
