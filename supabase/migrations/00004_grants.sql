-- Grant privileges on application tables to Supabase API roles.
--
-- SlideGuide connects with the service_role key and does not use
-- Row Level Security / auth (see backend/db/client.py). This project's
-- Postgres default privileges only grant `Dxtm` on newly created tables,
-- so without these explicit grants PostgREST returns 42501
-- ("permission denied for table ...") for every query. Granting DML to the
-- standard Supabase roles (and updating default privileges) makes the
-- documented `supabase start` / `supabase db reset` flow work out of the box.

grant usage on schema public to anon, authenticated, service_role;

grant all privileges on all tables in schema public
    to anon, authenticated, service_role;
grant all privileges on all sequences in schema public
    to anon, authenticated, service_role;
grant all privileges on all functions in schema public
    to anon, authenticated, service_role;

alter default privileges in schema public
    grant all on tables to anon, authenticated, service_role;
alter default privileges in schema public
    grant all on sequences to anon, authenticated, service_role;
alter default privileges in schema public
    grant all on functions to anon, authenticated, service_role;
