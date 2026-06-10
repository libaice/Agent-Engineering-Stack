create table documents (
  document_id text primary key,
  user_id text not null,
  org_id text not null,
  source text not null,
  file_type text not null,
  storage_uri text not null,
  acl jsonb default '{}',
  metadata jsonb default '{}',
  created_at timestamptz not null default now()
);