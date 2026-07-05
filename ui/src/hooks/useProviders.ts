import { useQuery } from "@tanstack/react-query"
import { getProviders } from "@/lib/api/client"

export function useProviders() {
  return useQuery({
    queryKey: ["providers"],
    queryFn: getProviders,
    staleTime: 60_000,
  })
}
