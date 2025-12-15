# TLDR

## - cURL 1 - Full Message Body

```sh
curl -X POST http://localhost:8080/ \
-H "Content-Type: application/json" \
-d '{ "data": "'$(echo -n '{"file_id":"68148e3f-ab0d-4ff8-9da4-9c72605f3602","filename":"jeopardy_qa_100_shorter.csv","gcs_bucket":"kalygo-kb-ingest-storage","gcs_file_path":"similarity_search/similarity_search/d8bf27a7-d1e7-4290-ae1b-efdbaa2ecb21/jeopardy_qa_100_shorter.csv","file_size":8053,"content_type":"text/csv","user_id":"1","namespace":"similarity_search","upload_timestamp":"2025-09-01T01:34:13.901871","processing_status":"pending","jwt":"<JWT_HERE>"}' | base64)'", "attributes": { "exampleAttribute": "exampleValue" } }'
```

##

eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0YWRAY21kbGFicy5pbyIsImlkIjoxLCJleHAiOjE3NjU4NjczODd9.1fhUCaVJdCtOUI6U2dEOVK5tA-8N77VUk1EOFeBs8oc
